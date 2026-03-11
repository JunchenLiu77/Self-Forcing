"""Focused WAN rollout repro for nested checkpointing.

Run:
  torchrun --nproc_per_node=2 tests/test_final.py --remat-n=2
  torchrun --nproc_per_node=2 tests/test_final.py --second-order --remat-n=2
"""
import functools
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint as ckpt
from wan.modules.causal_model import CausalWanAttentionBlock, rope_params
from utils.wan_cache import (
    build_crossattn_cache_entry,
    build_kv_cache_entry,
    clone_flat_cache_state,
    ensure_bool_tensor,
    flatten_cache_state,
    overwrite_cache_state_,
    unflatten_cache_state,
)


class TinyCausalWanModel(nn.Module):
    def __init__(
        self,
        dim=48,
        num_blocks=2,
        num_heads=2,
        ffn_dim=96,
        prime_dim=32,
        context_len=8,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.context_len = context_len
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    "t2v_cross_attn",
                    dim,
                    ffn_dim,
                    num_heads,
                    local_attn_size=-1,
                    sink_size=0,
                    qk_norm=True,
                    cross_attn_norm=False,
                    eps=1e-6,
                    prime_ffn_dim=prime_dim if idx == num_blocks - 1 else 0,
                )
                for idx in range(num_blocks)
            ]
        )
        self.gradient_checkpointing = True

        head_dim = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(64, head_dim - 4 * (head_dim // 6)),
                rope_params(64, 2 * (head_dim // 6)),
                rope_params(64, 2 * (head_dim // 6)),
            ],
            dim=1,
        )

    def clone_prime_params(self):
        fast = {}
        for idx, blk in enumerate(self.blocks):
            if blk.prime_ffn is not None:
                fast[idx] = {
                    name: parameter.clone()
                    for name, parameter in blk.prime_ffn.named_parameters()
                }
        return fast

    def get_prime_modules(self):
        return [block.prime_ffn for block in self.blocks if block.prime_ffn is not None]

    def forward_tokens(
        self,
        x,
        context,
        kv_cache,
        crossattn_cache,
        current_start,
        update_cache=True,
        fast_weights=None,
        detach_prime_input=False,
    ):
        batch_size, seq_len, _ = x.shape
        device = x.device
        grid_sizes = torch.tensor(
            [[1, 1, seq_len] for _ in range(batch_size)],
            device=device,
            dtype=torch.long,
        )
        seq_lens = torch.full(
            (batch_size,),
            seq_len,
            device=device,
            dtype=torch.long,
        )
        e = torch.zeros(batch_size, 1, 6, self.dim, device=device, dtype=x.dtype)

        def run_block(module, block_kwargs):
            def custom_forward(
                x,
                kv_k,
                kv_v,
                kv_global_end,
                kv_local_end,
                cross_k,
                cross_v,
                cross_is_init,
            ):
                x, kv_cache_entry, crossattn_cache_entry = module(
                    x,
                    kv_cache=build_kv_cache_entry(
                        kv_k, kv_v, kv_global_end, kv_local_end
                    ),
                    crossattn_cache=build_crossattn_cache_entry(
                        cross_k, cross_v, cross_is_init
                    ),
                    return_cache=True,
                    **block_kwargs,
                )
                return (
                    x,
                    kv_cache_entry["k"],
                    kv_cache_entry["v"],
                    kv_cache_entry["global_end_index"],
                    kv_cache_entry["local_end_index"],
                    crossattn_cache_entry["k"],
                    crossattn_cache_entry["v"],
                    ensure_bool_tensor(
                        crossattn_cache_entry["is_init"],
                        device=crossattn_cache_entry["k"].device,
                    ),
                )

            return custom_forward

        for block_index, block in enumerate(self.blocks):
            override = fast_weights.get(block_index) if fast_weights else None
            block_kwargs = {
                "e": e,
                "seq_lens": seq_lens,
                "grid_sizes": grid_sizes,
                "freqs": self.freqs.to(device),
                "context": context,
                "context_lens": None,
                "block_mask": None,
                "current_start": current_start,
                "cache_start": current_start,
                "update_cache": update_cache,
                "prime_ffn_override": override,
                "detach_prime_input": detach_prime_input,
            }
            kv_entry = kv_cache[block_index]
            cross_entry = crossattn_cache[block_index]
            block_inputs = (
                x,
                kv_entry["k"],
                kv_entry["v"],
                kv_entry["global_end_index"],
                kv_entry["local_end_index"],
                cross_entry["k"],
                cross_entry["v"],
                ensure_bool_tensor(
                    cross_entry["is_init"], device=cross_entry["k"].device
                ),
            )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                (
                    x,
                    kv_k,
                    kv_v,
                    kv_global_end,
                    kv_local_end,
                    cross_k,
                    cross_v,
                    cross_is_init,
                ) = ckpt(
                    run_block(block, block_kwargs),
                    *block_inputs,
                    use_reentrant=False,
                )
            else:
                x, kv_cache_entry, crossattn_cache_entry = block(
                    x,
                    kv_cache=kv_entry,
                    crossattn_cache=cross_entry,
                    return_cache=True,
                    **block_kwargs,
                )
                kv_k = kv_cache_entry["k"]
                kv_v = kv_cache_entry["v"]
                kv_global_end = kv_cache_entry["global_end_index"]
                kv_local_end = kv_cache_entry["local_end_index"]
                cross_k = crossattn_cache_entry["k"]
                cross_v = crossattn_cache_entry["v"]
                cross_is_init = ensure_bool_tensor(
                    crossattn_cache_entry["is_init"],
                    device=crossattn_cache_entry["k"].device,
                )
            kv_cache[block_index] = build_kv_cache_entry(
                kv_k,
                kv_v,
                kv_global_end,
                kv_local_end,
            )
            crossattn_cache[block_index] = build_crossattn_cache_entry(
                cross_k,
                cross_v,
                cross_is_init,
            )
        return x, kv_cache, crossattn_cache


class TinyWanWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = TinyCausalWanModel(**kwargs)

    def forward(
        self,
        x,
        context,
        kv_cache,
        crossattn_cache,
        current_start,
        update_cache=True,
        fast_weights=None,
        detach_prime_input=False,
        return_cache=False,
    ):
        output, new_kv_cache, new_crossattn_cache = self.model.forward_tokens(
            x,
            context,
            kv_cache,
            crossattn_cache,
            current_start=current_start,
            update_cache=update_cache,
            fast_weights=fast_weights,
            detach_prime_input=detach_prime_input,
        )
        if return_cache:
            return output, new_kv_cache, new_crossattn_cache
        overwrite_cache_state_(
            kv_cache,
            crossattn_cache,
            new_kv_cache,
            new_crossattn_cache,
        )
        return output


def _peel(module):
    if hasattr(module, "_fsdp_wrapped_module"):
        module = module._fsdp_wrapped_module
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module


def _init_tiny_wan_caches(model, batch_size, seq_len, device, dtype):
    head_dim = model.dim // model.num_heads
    kv_cache_size = seq_len * 8
    kv_cache = []
    crossattn_cache = []
    for _ in model.blocks:
        kv_cache.append(
            build_kv_cache_entry(
                torch.zeros(
                    batch_size,
                    kv_cache_size,
                    model.num_heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                ),
                torch.zeros(
                    batch_size,
                    kv_cache_size,
                    model.num_heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                ),
                torch.tensor([0], device=device, dtype=torch.long),
                torch.tensor([0], device=device, dtype=torch.long),
            )
        )
        crossattn_cache.append(
            build_crossattn_cache_entry(
                torch.zeros(
                    batch_size,
                    model.context_len,
                    model.num_heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                ),
                torch.zeros(
                    batch_size,
                    model.context_len,
                    model.num_heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                ),
                torch.tensor([False], device=device, dtype=torch.bool),
            )
        )
    return kv_cache, crossattn_cache


def _compute_tiny_wan_inner_loss(
    wrapper,
    pseudo_gt,
    noise,
    context,
    kv_cache,
    crossattn_cache,
    current_start,
    fast_weights,
):
    with torch.no_grad():
        noisy = pseudo_gt + 0.5 * noise
        flow_pred, _, _ = wrapper(
            noisy.detach(),
            context,
            kv_cache,
            crossattn_cache,
            current_start=current_start,
            update_cache=False,
            fast_weights=fast_weights,
            detach_prime_input=True,
            return_cache=True,
        )
    target = noise - pseudo_gt
    return F.mse_loss(flow_pred, target)


def _flatten_fast_params(fast_params):
    keys = []
    values = []
    for block_idx in sorted(fast_params):
        for name in sorted(fast_params[block_idx]):
            keys.append((block_idx, name))
            values.append(fast_params[block_idx][name])
    return keys, tuple(values)


def _unflatten_fast_params(keys, values):
    fast_params = {}
    for (block_idx, name), value in zip(keys, values):
        fast_params.setdefault(block_idx, {})[name] = value
    return fast_params


def inner_step(fast_params, inner_loss, lr, create_graph=False):
    all_tensors, keys = [], []
    for block_idx in sorted(fast_params):
        for name in sorted(fast_params[block_idx]):
            keys.append((block_idx, name))
            all_tensors.append(fast_params[block_idx][name])
    grads = torch.autograd.grad(
        inner_loss,
        all_tensors,
        create_graph=create_graph,
        allow_unused=True,
    )
    grads = [
        grad if grad is not None else torch.zeros_like(tensor)
        for grad, tensor in zip(grads, all_tensors)
    ]
    new_fast = {}
    for (block_idx, name), grad in zip(keys, grads):
        new_fast.setdefault(block_idx, {})[name] = (
            fast_params[block_idx][name] - lr * grad
        )
    return new_fast


def _all_reduce_prime_grads(backbone):
    for prime_module in backbone.get_prime_modules():
        for parameter in prime_module.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def _grad_vector(module):
    grads = [
        parameter.grad.detach().reshape(-1).float().clone()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    assert grads, "Expected at least one gradient tensor"
    return torch.cat(grads)


def _parse_remat_n():
    for arg in sys.argv:
        if arg.startswith("--remat-n="):
            return int(arg.split("=", 1)[1])
    return None


def _build_tiny_wan_fsdp_wrapper(
    device,
    dim=48,
    num_blocks=2,
    num_heads=2,
    ffn_dim=96,
    prime_dim=32,
    context_len=8,
):
    torch.manual_seed(123)
    wrapper = TinyWanWrapper(
        dim=dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        prime_dim=prime_dim,
        context_len=context_len,
    ).to(device)
    prime_ignored = [
        block.prime_ffn for block in wrapper.model.blocks if block.prime_ffn is not None
    ]
    auto_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={CausalWanAttentionBlock},
    )
    wrapper = FSDP(
        wrapper,
        auto_wrap_policy=auto_policy,
        ignored_modules=prime_ignored,
        device_id=device,
        use_orig_params=True,
    )
    for module in prime_ignored:
        module.to(device=device, dtype=torch.float32)
    return wrapper


def _run_tiny_wan_rollout(
    wrapper,
    x_blocks,
    noise_blocks,
    context,
    inner_lr,
    remat_n,
    second_order=False,
):
    backbone = _peel(_peel(wrapper).model)
    kv_cache, crossattn_cache = _init_tiny_wan_caches(
        backbone,
        batch_size=x_blocks.shape[1],
        seq_len=x_blocks.shape[2],
        device=x_blocks.device,
        dtype=x_blocks.dtype,
    )
    fast_params = backbone.clone_prime_params()
    if remat_n <= 0:
        raise ValueError("WAN rollout repro expects remat_n > 0")

    assert x_blocks.shape[0] % remat_n == 0, (
        f"vblocks ({x_blocks.shape[0]}) must be divisible by remat_n ({remat_n})"
    )
    fast_keys, fast_values = _flatten_fast_params(fast_params)
    num_fast = len(fast_values)
    n_segments = x_blocks.shape[0] // remat_n
    x_segments = x_blocks.reshape(n_segments, remat_n, *x_blocks.shape[1:])
    noise_segments = noise_blocks.reshape(
        n_segments, remat_n, *noise_blocks.shape[1:]
    )
    outputs = []
    current_start = 0

    for seg_idx in range(n_segments):
        flat_cache = flatten_cache_state(kv_cache, crossattn_cache)
        x_segment = x_segments[seg_idx]
        noise_segment = noise_segments[seg_idx]

        def run_segment(
            flat_fast_values,
            *flat_cache_values,
            current_start=current_start,
            x_segment=x_segment,
            noise_segment=noise_segment,
        ):
            flat_cache_values = clone_flat_cache_state(flat_cache_values)
            local_kv_cache, local_crossattn_cache = unflatten_cache_state(
                flat_cache_values,
                num_kv_blocks=len(backbone.blocks),
                num_crossattn_blocks=len(backbone.blocks),
            )
            local_fast_params = _unflatten_fast_params(fast_keys, flat_fast_values)
            local_outputs = []
            for segment_index in range(x_segment.shape[0]):
                block_start = current_start + segment_index * x_blocks.shape[2]
                fw = local_fast_params if local_fast_params else None
                with torch.enable_grad():
                    out, local_kv_cache, local_crossattn_cache = wrapper(
                        x_segment[segment_index],
                        context,
                        local_kv_cache,
                        local_crossattn_cache,
                        current_start=block_start,
                        update_cache=True,
                        fast_weights=fw,
                        return_cache=True,
                    )
                local_outputs.append(out)

                inner_loss = _compute_tiny_wan_inner_loss(
                    wrapper,
                    out.detach(),
                    noise_segment[segment_index],
                    context,
                    local_kv_cache,
                    local_crossattn_cache,
                    block_start,
                    fw,
                )
                local_fast_params = inner_step(
                    local_fast_params,
                    inner_loss,
                    inner_lr,
                    create_graph=second_order,
                )

                with torch.no_grad():
                    _, local_kv_cache, local_crossattn_cache = wrapper(
                        out.detach(),
                        context,
                        local_kv_cache,
                        local_crossattn_cache,
                        current_start=block_start,
                        update_cache=True,
                        return_cache=True,
                    )

            _, new_fast_values = _flatten_fast_params(local_fast_params)
            return (
                new_fast_values
                + flatten_cache_state(local_kv_cache, local_crossattn_cache)
                + (torch.stack(local_outputs),)
            )

        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            result = ckpt(
                run_segment,
                fast_values,
                *flat_cache,
                use_reentrant=False,
            )
        num_cache = len(flat_cache)
        fast_values = result[:num_fast]
        kv_cache, crossattn_cache = unflatten_cache_state(
            result[num_fast:num_fast + num_cache],
            num_kv_blocks=len(backbone.blocks),
            num_crossattn_blocks=len(backbone.blocks),
        )
        outputs.append(result[num_fast + num_cache])
        current_start += remat_n * x_blocks.shape[2]

    return torch.cat(outputs, dim=0)


def _collect_tiny_wan_loss_and_grads(
    wrapper,
    x_blocks,
    noise_blocks,
    context,
    inner_lr,
    remat_n,
    world,
    second_order=False,
):
    wrapper.zero_grad(set_to_none=True)
    outputs = _run_tiny_wan_rollout(
        wrapper,
        x_blocks,
        noise_blocks,
        context,
        inner_lr,
        remat_n,
        second_order=second_order,
    )
    outer_loss = outputs.sum() / world
    outer_loss.backward()
    backbone = _peel(_peel(wrapper).model)
    _all_reduce_prime_grads(backbone)
    grad_vector = _grad_vector(wrapper)
    assert torch.isfinite(grad_vector).all()


def run_wan_rollout_repro(
    device,
    world,
    remat_n,
    second_order=False,
):
    if remat_n <= 0:
        raise ValueError("repro-wan-rollout expects remat_n > 0")

    dim, num_blocks, num_heads = 48, 2, 2
    seq, bs, vblocks = 4, 1, 4
    inner_lr = 0.01

    torch.manual_seed(5000 + dist.get_rank())
    torch.cuda.manual_seed(5000 + dist.get_rank())

    wrapper = _build_tiny_wan_fsdp_wrapper(
        dim=dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        ffn_dim=dim * 2,
        prime_dim=32,
        context_len=8,
        device=device,
    )
    x_blocks = torch.randn(vblocks, bs, seq, dim, device=device)
    noise_blocks = torch.randn(vblocks, bs, seq, dim, device=device)
    context = torch.randn(
        bs, _peel(_peel(wrapper).model).context_len, dim, device=device
    )
    _collect_tiny_wan_loss_and_grads(
        wrapper,
        x_blocks,
        noise_blocks,
        context,
        inner_lr,
        remat_n=remat_n,
        world=world,
        second_order=second_order,
    )

    if dist.get_rank() == 0:
        print("Explicit-cache WAN rollout passed nested checkpoint repro.")


def main():
    unsupported_flags = {
        "--naive",
        "--mixed-precision",
        "--repro-nested",
        "--repro-rollout",
        "--checkpoint-debug",
        "--repro-wan-rollout",
    }
    unsupported_prefixes = (
        "--nested-policy=",
        "--rollout-inner=",
    )
    for arg in sys.argv[1:]:
        if arg in unsupported_flags or any(
            arg.startswith(prefix) for prefix in unsupported_prefixes
        ):
            raise ValueError(
                "tests/test_final.py now only supports the final WAN rollout "
                "repro path. Use --remat-n=<n> and optional --second-order."
            )

    second_order = "--second-order" in sys.argv
    requested_remat_n = _parse_remat_n()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    train_remat_n = 2 if requested_remat_n is None else requested_remat_n

    if rank == 0:
        print(
            f"world={world}, train_remat_n={train_remat_n}, "
            f"second_order={second_order}"
        )

    run_wan_rollout_repro(
        device,
        world,
        remat_n=train_remat_n,
        second_order=second_order,
    )

    dist.destroy_process_group()
    if rank == 0:
        print("Wan rollout repro path completed without failure.")


if __name__ == "__main__":
    main()
