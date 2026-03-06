"""Meta-learning: functional AdamW inner loop, AdamW outer loop."""
import math

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


def adamw_step(
    params,
    grads,
    state,
    *,
    lr,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-2,
):
    """Apply an AdamW update without mutating module parameters."""
    beta1, beta2 = betas
    new_params = {}
    new_state = {}

    for (name, param), grad in zip(params.items(), grads):
        param_state = state.get(name)
        if param_state is None:
            step = 0
            exp_avg = torch.zeros_like(param)
            exp_avg_sq = torch.zeros_like(param)
        else:
            step = param_state["step"]
            exp_avg = param_state["exp_avg"]
            exp_avg_sq = param_state["exp_avg_sq"]

        step += 1
        exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad.square()

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + eps
        step_size = lr / bias_correction1

        decayed_param = param * (1 - lr * weight_decay)
        new_params[name] = decayed_param - step_size * exp_avg / denom
        new_state[name] = {
            "step": step,
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
        }

    return new_params, new_state


@pytest.mark.parametrize("inner_lr", [1e-3])
@pytest.mark.parametrize("outer_lr", [1e-3])
@pytest.mark.parametrize("inner_iters", [2])
@pytest.mark.parametrize("inner_wd", [1e-2])
@pytest.mark.parametrize("outer_wd", [1e-2])
def test(inner_lr, outer_lr, inner_iters, inner_wd, outer_wd):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    model = Model().to(device)
    slow_params = tuple(model.slow.parameters())
    fast_params = tuple(model.fast.parameters())
    outer_opt = torch.optim.AdamW(model.parameters(), lr=outer_lr, weight_decay=outer_wd)

    # data
    x = torch.randn(4, 4, device=device)
    y = torch.randint(0, 4, (4,), device=device)

    cur_fast = {
        n: p.clone()
        for n, p in model.named_parameters()
        if n.startswith("fast")
    }
    cur_fast_state = {}

    outer_opt.zero_grad()
    inner_losses = []

    # Inner loop: update only the fast weights with AdamW optimizer.
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
        cur_fast, cur_fast_state = adamw_step(
            cur_fast,
            inner_grads,
            cur_fast_state,
            lr=inner_lr,
            weight_decay=inner_wd,
        )

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