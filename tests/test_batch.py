"""Meta-learning: batched tasks with vmapped inner loop."""
import torch
import torch.nn as nn
from torch.func import functional_call, vmap
import pytest


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.slow = nn.Linear(4, 8)
        self.fast = nn.Linear(8, 4)

    def forward(self, x):
        return self.fast(self.slow(x))


@pytest.mark.parametrize("inner_lr", [1e-3])
@pytest.mark.parametrize("outer_lr", [1e-3])
@pytest.mark.parametrize("inner_iters", [2])
@pytest.mark.parametrize("bs", [2])
def test(inner_lr, outer_lr, inner_iters, bs):
    samples_per_task = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    model = Model().to(device)
    slow_params = tuple(model.slow.parameters())
    fast_params = tuple(model.fast.parameters())
    outer_opt = torch.optim.SGD(model.parameters(), lr=outer_lr)

    # Per-task data: [bs, samples_per_task, ...]
    xs = torch.randn(bs, samples_per_task, 4, device=device)
    ys = torch.randint(0, 4, (bs, samples_per_task), device=device)

    fast_names = [n for n, _ in model.named_parameters() if n.startswith("fast")]

    # Each task starts from the same initial fast weights.
    # Shape per tensor: [bs, *param_shape].
    # clone() keeps a grad path back to the original nn.Parameters.
    batched_fast = tuple(
        p.unsqueeze(0).expand(bs, *p.shape).clone()
        for n, p in model.named_parameters()
        if n.startswith("fast")
    )

    def per_task_loss(fast_values, x, y):
        params = {n: p for n, p in model.named_parameters()
                  if not n.startswith("fast")}
        for name, val in zip(fast_names, fast_values):
            params[name] = val
        logits = functional_call(model, params, (x,))
        return nn.functional.cross_entropy(logits, y)

    vmapped_loss = vmap(
        per_task_loss,
        in_dims=(tuple(0 for _ in fast_names), 0, 0),
    )

    outer_opt.zero_grad()
    all_inner_losses = []

    for _ in range(inner_iters):
        # Vectorised per-task losses: [bs]
        losses = vmapped_loss(batched_fast, xs, ys)
        all_inner_losses.append(losses)

        # Because each task's loss depends only on its own fast-weight slice,
        # grad(sum_of_losses) gives the per-task gradients stacked along dim 0.
        grads = torch.autograd.grad(
            losses.sum(), batched_fast, create_graph=True,
        )

        # Functional SGD inner update (vectorised across tasks, no loop)
        batched_fast = tuple(
            p - inner_lr * g for p, g in zip(batched_fast, grads)
        )

    # Outer loss: mean over tasks and inner steps
    outer_loss = torch.stack(all_inner_losses).mean()

    outer_grads = torch.autograd.grad(
        outer_loss,
        slow_params + fast_params,
        create_graph=True,
        retain_graph=True,
    )

    for grad in outer_grads:
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().max().item() > 0

    slow_grad_norm = sum(g.abs().sum() for g in outer_grads[:len(slow_params)])
    fast_grad_norm = sum(g.abs().sum() for g in outer_grads[len(slow_params):])
    assert slow_grad_norm.item() > 0
    assert fast_grad_norm.item() > 0

    # Higher-order: outer gradients should back-prop into the init fast weights.
    higher_order_loss = sum(g.pow(2).sum() for g in outer_grads)
    higher_order_grads = torch.autograd.grad(
        higher_order_loss, fast_params, retain_graph=True,
    )

    for grad in higher_order_grads:
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum().item() > 0

    # Actual outer optimizer step should change both slow and fast params.
    outer_opt.zero_grad()
    slow_before = [p.detach().clone() for p in slow_params]
    fast_before = [p.detach().clone() for p in fast_params]
    outer_loss.backward()
    outer_opt.step()

    for before, after in zip(slow_before, slow_params):
        assert not torch.allclose(before, after.detach())
    for before, after in zip(fast_before, fast_params):
        assert not torch.allclose(before, after.detach())
