"""Meta-learning: FSDP batched tasks with vmapped inner loop.

Run:  torchrun --nproc_per_node=2 tests/test_batch_fsdp.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.func import vmap


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.slow = nn.Linear(4, 8)
        self.fast = nn.Linear(8, 4)

    def forward(self, x):
        return self.fast(self.slow(x))


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # hyper-params
    inner_lr = 1e-3
    outer_lr = 1e-3
    inner_iters = 2
    local_bs = 2
    samples_per_task = 4

    # model (identical init on every rank)
    torch.manual_seed(42)
    model = Model().to(device)
    # FSDP-wrap only the slow sub-module; fast stays replicated.
    # In this way each rank would have their own fast weights.
    model.slow = FSDP(model.slow, device_id=device)

    fast_params = [model.fast.weight, model.fast.bias]
    outer_opt = torch.optim.SGD(model.parameters(), lr=outer_lr)

    # per-rank data (seeded by rank so each rank gets unique data)
    torch.manual_seed(1000 + rank)
    torch.cuda.manual_seed(1000 + rank)
    xs = torch.randn(local_bs, samples_per_task, 4, device=device)
    ys = torch.randint(0, 4, (local_bs, samples_per_task), device=device)

    # per-rank fast-weight copies: [local_bs, *param_shape]
    # clone() keeps the gradient path back to model.fast.{weight,bias}
    fast_w = model.fast.weight.unsqueeze(0).expand(local_bs, -1, -1).clone()
    fast_b = model.fast.bias.unsqueeze(0).expand(local_bs, -1).clone()

    def per_task_fast_loss(fw, fb, h, y):
        return F.cross_entropy(F.linear(h, fw, fb), y)

    vmapped_fast_loss = vmap(per_task_fast_loss)

    outer_opt.zero_grad()
    all_inner_losses = []

    for _ in range(inner_iters):
        # Slow forward through FSDP (handles all-gather internally).
        # Flatten local_bs * samples_per_task into a single batch.
        hidden = model.slow(
            xs.reshape(-1, 4)
        ).reshape(local_bs, samples_per_task, 8)

        # Vectorised fast forward + CE loss per local task
        losses = vmapped_fast_loss(fast_w, fast_b, hidden, ys)  # [local_bs]
        all_inner_losses.append(losses)

        # Per-task fast-weight grads (graph kept for outer backprop)
        gw, gb = torch.autograd.grad(
            losses.sum(), (fast_w, fast_b), create_graph=True,
        )

        # Functional SGD inner step (vectorised across tasks, no loop)
        fast_w = fast_w - inner_lr * gw
        fast_b = fast_b - inner_lr * gb

    # outer loss
    # Divide by world_size so that FSDP's gradient-sum (reduce-scatter)
    # across ranks yields the gradient of the true global mean.
    outer_loss = torch.stack(all_inner_losses).mean() / world_size
    outer_loss.backward()

    # FSDP already synced slow-param grads via reduce-scatter.
    # Manually all-reduce fast-param grads (replicated, not FSDP-managed).
    for p in fast_params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

    for p in fast_params:
        assert p.grad is not None, "fast param missing gradient"
        assert torch.isfinite(p.grad).all(), "fast param grad not finite"
        assert p.grad.abs().max().item() > 0, "fast param grad is zero"

    # Snapshot both param groups before the step.
    with FSDP.summon_full_params(model.slow):
        slow_before = {n: p.detach().clone()
                       for n, p in model.slow.named_parameters()}
    fast_before = [p.detach().clone() for p in fast_params]

    outer_opt.step()

    # Both groups should have changed.
    with FSDP.summon_full_params(model.slow):
        for n, p in model.slow.named_parameters():
            assert not torch.allclose(slow_before[n], p.detach()), \
                f"slow param {n} unchanged after step"

    for before, p in zip(fast_before, fast_params):
        assert not torch.allclose(before, p.detach()), "fast param unchanged"

    # Fast params must agree across all ranks (same init + same gradient).
    for p in fast_params:
        gathered = [torch.empty_like(p) for _ in range(world_size)]
        dist.all_gather(gathered, p.detach())
        for i in range(1, world_size):
            assert torch.allclose(gathered[0], gathered[i]), \
                f"fast param diverged between rank 0 and rank {i}"

    dist.destroy_process_group()
    if rank == 0:
        print("PASSED")


if __name__ == "__main__":
    main()
