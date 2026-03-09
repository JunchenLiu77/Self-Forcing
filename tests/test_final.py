"""Reproduce FSDP + gradient-checkpointing + inner autograd.grad crash.

Strategies:
  --naive     inner loop through FSDP wrapper with enable_grad (FAILS)
  --nograd    inner loop under no_grad; PrimeFFN uses enable_grad internally (DEFAULT, PASSES)

Run:  torchrun --nproc_per_node=2 tests/test_final.py --naive
      torchrun --nproc_per_node=2 tests/test_final.py
      torchrun --nproc_per_node=2 tests/test_final.py --mixed-precision
"""
import functools
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint as ckpt


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PrimeFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, prime_dim=0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.prime_ffn = PrimeFFN(dim, prime_dim) if prime_dim > 0 else None
        self._prime_out = None

    def forward(self, x, prime_override=None, detach_prime_input=False):
        x = x + self.fc2(F.gelu(self.fc1(x)))
        if self.prime_ffn is not None:
            if prime_override is not None:
                if detach_prime_input:
                    with torch.enable_grad():
                        prime_out = functional_call(
                            self.prime_ffn, prime_override, (x.detach(),))
                        self._prime_out = prime_out
                        x = x + prime_out
                else:
                    x = x + functional_call(
                        self.prime_ffn, prime_override, (x,))
            else:
                x = x + self.prime_ffn(x)
        return x


class Backbone(nn.Module):
    def __init__(self, dim=64, num_blocks=6, suffix_len=2, prime_dim=32):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, prime_dim if i >= num_blocks - suffix_len else 0)
            for i in range(num_blocks)
        ])
        self.head = nn.Linear(dim, dim)
        self.gradient_checkpointing = True

    def forward(self, x, fast_weights=None, detach_prime_input=False):
        for idx, block in enumerate(self.blocks):
            override = fast_weights.get(idx) if fast_weights else None
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = ckpt(
                    block, x, override, detach_prime_input,
                    use_reentrant=False)
            else:
                x = block(x, prime_override=override,
                          detach_prime_input=detach_prime_input)
        if detach_prime_input:
            with torch.enable_grad():
                return self.head(x)
        return self.head(x)

    def get_prime_modules(self):
        return [b.prime_ffn for b in self.blocks if b.prime_ffn is not None]

    def clone_prime_params(self):
        fast = {}
        for idx, blk in enumerate(self.blocks):
            if blk.prime_ffn is not None:
                fast[idx] = {
                    n: p.clone()
                    for n, p in blk.prime_ffn.named_parameters()
                }
        return fast


class Wrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = Backbone(**kwargs)

    def forward(self, x, fast_weights=None, detach_prime_input=False):
        return self.model(x, fast_weights=fast_weights,
                          detach_prime_input=detach_prime_input)


# ---------------------------------------------------------------------------
# Inner-loop helpers
# ---------------------------------------------------------------------------

def _peel(m):
    if hasattr(m, "_fsdp_wrapped_module"):
        m = m._fsdp_wrapped_module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def inner_step(fast_params, inner_loss, lr):
    all_tensors, keys = [], []
    for bi in sorted(fast_params):
        for n in sorted(fast_params[bi]):
            keys.append((bi, n))
            all_tensors.append(fast_params[bi][n])
    grads = torch.autograd.grad(
        inner_loss, all_tensors, create_graph=False, allow_unused=True)
    grads = [g if g is not None else torch.zeros_like(t)
             for g, t in zip(grads, all_tensors)]
    new = {}
    for (b, n), g in zip(keys, grads):
        new.setdefault(b, {})[n] = fast_params[b][n] - lr * g
    return new


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    strategy = "nograd"
    if "--naive" in sys.argv:
        strategy = "naive"
    use_mp = "--mixed-precision" in sys.argv

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"strategy={strategy}, mp={use_mp}, world={world}")

    DIM, NUM_BLOCKS, SUFFIX, PRIME_DIM = 64, 6, 2, 32
    INNER_LR, OUTER_LR, SEQ, BS, VBLOCKS = 0.01, 0.01, 16, 2, 4

    torch.manual_seed(42)
    wrapper = Wrapper(
        dim=DIM, num_blocks=NUM_BLOCKS,
        suffix_len=SUFFIX, prime_dim=PRIME_DIM,
    ).to(device)

    fsdp_kw = dict(device_id=device)
    if use_mp:
        fsdp_kw["mixed_precision"] = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

    prime_ignored = wrapper.model.get_prime_modules()
    auto_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100)
    wrapper = FSDP(
        wrapper, auto_wrap_policy=auto_policy,
        ignored_modules=prime_ignored, **fsdp_kw)
    for m in prime_ignored:
        m.to(device=device, dtype=torch.bfloat16 if use_mp else torch.float32)

    backbone = _peel(_peel(wrapper).model)
    opt = torch.optim.SGD(wrapper.parameters(), lr=OUTER_LR)

    torch.manual_seed(1000 + rank)
    torch.cuda.manual_seed(1000 + rank)
    loss_history = []

    for step in range(2):
        opt.zero_grad()
        fast_params = backbone.clone_prime_params()
        exit_outputs = []

        # Wrap the entire block loop in no_grad to simulate the critic
        # step, where inference_with_trajectory is called under no_grad.
        # The exit step uses set_grad_enabled(True) just like the real
        # pipeline.  The inner loop's _compute_inner_loss uses
        # @torch.enable_grad() — without it, inner_loss would have
        # requires_grad=False and autograd.grad would fail.
        with torch.no_grad():
            for vi in range(VBLOCKS):
                x_block = torch.randn(BS, SEQ, DIM, device=device)
                fw = fast_params

                # ---- exit step (grad enabled, like the real pipeline) ----
                with torch.enable_grad():
                    out = wrapper(x_block, fast_weights=fw)
                exit_outputs.append(out)

                # ---- inner loop ----
                pseudo_gt = out.detach()
                noise = torch.randn_like(pseudo_gt)
                noisy = pseudo_gt + 0.5 * noise

                if strategy == "naive":
                    with torch.enable_grad():
                        pred = wrapper(noisy, fast_weights=fw)
                    inner_loss = F.mse_loss(pred, noise - pseudo_gt)
                    fast_params = inner_step(
                        fast_params, inner_loss, INNER_LR)

                elif strategy == "nograd":
                    pred = wrapper(noisy, fast_weights=fw,
                                   detach_prime_input=True)
                    # @torch.enable_grad() in the real pipeline
                    with torch.enable_grad():
                        inner_loss = F.mse_loss(pred, noise - pseudo_gt)
                        fast_params = inner_step(
                            fast_params, inner_loss, INNER_LR)
                    del inner_loss, pred
                    torch.cuda.empty_cache()

                # ---- cache update (already under no_grad) ----
                _ = wrapper(pseudo_gt, fast_weights=fw)

        outer_loss = sum(o.sum() for o in exit_outputs) / world
        outer_loss.backward()

        for p in backbone.get_prime_modules()[0].parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

        grad_norm = torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
        assert torch.isfinite(grad_norm), f"step {step}: grad not finite"
        opt.step()
        loss_history.append(outer_loss.item() * world)
        if rank == 0:
            print(f"  step {step}: loss={loss_history[-1]:.4f}")

    dist.destroy_process_group()
    if rank == 0:
        print("PASSED")


if __name__ == "__main__":
    main()
