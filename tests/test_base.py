"""Meta-learning: basic inner/outer loop with functional SGD."""
import torch
import torch.nn as nn
from torch.func import functional_call
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
def test(inner_lr, outer_lr, inner_iters):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    
    model = Model().to(device)
    slow_params = tuple(model.slow.parameters())
    fast_params = tuple(model.fast.parameters())
    outer_opt = torch.optim.SGD(model.parameters(), lr=outer_lr)

    # data
    x = torch.randn(4, 4, device=device)
    y = torch.randint(0, 4, (4,), device=device)

    cur_fast = {
        n: p.clone()
        for n, p in model.named_parameters()
        if n.startswith("fast")
    }

    outer_opt.zero_grad()
    inner_losses = []

    # Inner loop: update only the fast weights with simple SGD steps.
    for _ in range(inner_iters):
        all_params = dict(model.named_parameters())
        all_params.update(cur_fast)

        inner_logits = functional_call(model, all_params, (x,))
        inner_loss = nn.functional.cross_entropy(inner_logits, y)
        inner_losses.append(inner_loss)
        inner_grads = torch.autograd.grad(
            inner_loss,
            tuple(cur_fast.values()),
            create_graph=True,
        )
        # use simplest SGD update rule for inner loop updating
        cur_fast = {
            n: p - inner_lr * grad
            for (n, p), grad in zip(cur_fast.items(), inner_grads)
        }

    # Outer loop: optimize both slow and fast weights using the
    # average loss over the inner-loop trajectory.
    outer_loss = torch.stack(inner_losses).mean()

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

    slow_grad_norm = sum(grad.abs().sum() for grad in outer_grads[: len(slow_params)])
    fast_grad_norm = sum(grad.abs().sum() for grad in outer_grads[len(slow_params) :])
    assert slow_grad_norm.item() > 0
    assert fast_grad_norm.item() > 0

    # A second-order objective over the outer gradients should still backprop
    # into the initial fast weights.
    higher_order_loss = sum(grad.pow(2).sum() for grad in outer_grads)
    higher_order_grads = torch.autograd.grad(
        higher_order_loss,
        fast_params,
        retain_graph=True,
    )

    for grad in higher_order_grads:
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum().item() > 0

    # Run a real outer optimizer step and verify both parameter groups change.
    outer_opt.zero_grad()
    slow_before = [p.detach().clone() for p in slow_params]
    fast_before = [p.detach().clone() for p in fast_params]
    outer_loss.backward()
    outer_opt.step()

    for before, after in zip(slow_before, slow_params):
        assert not torch.allclose(before, after.detach())
    for before, after in zip(fast_before, fast_params):
        assert not torch.allclose(before, after.detach())