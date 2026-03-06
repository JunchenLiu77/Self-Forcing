"""Meta-learning: FSDP + batched + multi-outer + sequence checkpointing.

Run:  torchrun --nproc_per_node=2 tests/test_final.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.func import vmap
from torch.utils.checkpoint import checkpoint as ckpt


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.slow = nn.Linear(4, 8)
        self.fast = nn.Linear(8, 4)

    def forward(self, x):
        return self.fast(self.slow(x))


# ---- vmapped fast loss ----

def _per_task_loss(fw, fb, h, y):
    return F.cross_entropy(F.linear(h, fw, fb), y)


_vmapped_loss = vmap(_per_task_loss)


# ---- single chunk step ----

def _chunk_step(model, fast_w, fast_b, x_chunk, y_chunk, inner_lr):
    local_bs = fast_w.shape[0]
    hidden = model.slow(
        x_chunk.reshape(-1, x_chunk.shape[-1])
    ).reshape(local_bs, -1, 8)

    losses = _vmapped_loss(fast_w, fast_b, hidden, y_chunk)

    gw, gb = torch.autograd.grad(
        losses.sum(), (fast_w, fast_b), create_graph=True,
    )
    fast_w = fast_w - inner_lr * gw
    fast_b = fast_b - inner_lr * gb
    return fast_w, fast_b, losses


# ---- remat segment ----

def _segment_body(fast_w, fast_b, x_seg, y_seg, model, inner_lr):
    seg_losses = []
    for i in range(x_seg.shape[0]):
        fast_w, fast_b, losses = _chunk_step(
            model, fast_w, fast_b, x_seg[i], y_seg[i], inner_lr,
        )
        seg_losses.append(losses)
    return fast_w, fast_b, torch.stack(seg_losses)


def run_inner_loop(model, fast_w, fast_b, x_chunks, y_chunks,
                   inner_lr, remat_n):
    num_chunks = x_chunks.shape[0]
    n_segments = num_chunks // remat_n

    x_segs = x_chunks.reshape(n_segments, remat_n, *x_chunks.shape[1:])
    y_segs = y_chunks.reshape(n_segments, remat_n, *y_chunks.shape[1:])

    all_losses = []
    for seg_idx in range(n_segments):
        fast_w, fast_b, seg_losses = ckpt(
            _segment_body,
            fast_w, fast_b,
            x_segs[seg_idx], y_segs[seg_idx],
            model, inner_lr,
            use_reentrant=False,
        )
        all_losses.append(seg_losses)

    return fast_w, fast_b, torch.cat(all_losses, dim=0)


# ---- main ----

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # ---- config ----
    inner_lr = 1e-2
    outer_lr = 1e-2
    local_bs = 2
    num_chunks = 4
    chunk_len = 2
    remat_n = 2
    outer_iters = 2

    # ---- model (identical init on every rank) ----
    torch.manual_seed(42)
    model = Model().to(device)
    model.slow = FSDP(model.slow, device_id=device)

    fast_params = [model.fast.weight, model.fast.bias]
    outer_opt = torch.optim.SGD(model.parameters(), lr=outer_lr)

    # Snapshot initial values before training.
    with FSDP.summon_full_params(model.slow):
        slow_init = {n: p.detach().clone()
                     for n, p in model.slow.named_parameters()}
    fast_init = [p.detach().clone() for p in fast_params]

    # ---- per-rank data (seeded by rank) ----
    torch.manual_seed(1000 + rank)
    torch.cuda.manual_seed(1000 + rank)
    seq_len = num_chunks * chunk_len
    xs = torch.randn(local_bs, seq_len, 4, device=device)
    ys = torch.randint(0, 4, (local_bs, seq_len), device=device)

    # Reshape to [num_chunks, local_bs, chunk_len, ...].
    x_chunks = (xs.reshape(local_bs, num_chunks, chunk_len, 4)
                  .permute(1, 0, 2, 3).contiguous())
    y_chunks = (ys.reshape(local_bs, num_chunks, chunk_len)
                  .permute(1, 0, 2).contiguous())

    # ---- multi-outer loop ----
    loss_history = []

    for outer_step in range(outer_iters):
        # Fresh per-rank fast-weight copies from the (updated) model.
        fast_w = model.fast.weight.unsqueeze(0).expand(local_bs, -1, -1).clone()
        fast_b = model.fast.bias.unsqueeze(0).expand(local_bs, -1).clone()

        outer_opt.zero_grad()

        _, _, chunk_losses = run_inner_loop(
            model, fast_w, fast_b, x_chunks, y_chunks, inner_lr, remat_n,
        )

        # Scale by 1/world_size so FSDP's gradient-sum = global average.
        outer_loss = chunk_losses.mean() / world_size
        outer_loss.backward()

        # FSDP synced slow-param grads.  Manually sync fast-param grads.
        for p in fast_params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

        # Check gradients before the step.
        for p in fast_params:
            assert p.grad is not None, \
                f"step {outer_step}: fast param missing gradient"
            assert torch.isfinite(p.grad).all(), \
                f"step {outer_step}: fast param grad not finite"
            assert p.grad.abs().max().item() > 0, \
                f"step {outer_step}: fast param grad is zero"

        outer_opt.step()

        loss_val = outer_loss.item() * world_size  # undo scaling for logging
        loss_history.append(loss_val)
        assert torch.isfinite(torch.tensor(loss_val)), \
            f"step {outer_step}: loss not finite ({loss_val})"

    # ---- post-training assertions ----
    # Both param groups should have moved from init.
    with FSDP.summon_full_params(model.slow):
        for n, p in model.slow.named_parameters():
            assert not torch.allclose(slow_init[n], p.detach()), \
                f"slow param {n} unchanged after training"

    for init_p, p in zip(fast_init, fast_params):
        assert not torch.allclose(init_p, p.detach()), \
            "fast param unchanged after training"

    # Fast params must agree across ranks.
    for p in fast_params:
        gathered = [torch.empty_like(p) for _ in range(world_size)]
        dist.all_gather(gathered, p.detach())
        for i in range(1, world_size):
            assert torch.allclose(gathered[0], gathered[i]), \
                f"fast param diverged between rank 0 and rank {i}"

    # Loss should have decreased over outer iterations.
    assert loss_history[-1] < loss_history[0], (
        f"loss did not decrease: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}"
    )

    dist.destroy_process_group()
    if rank == 0:
        print("PASSED")


if __name__ == "__main__":
    main()
