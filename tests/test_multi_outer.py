"""Meta-learning: multiple outer optimisation steps with convergence check."""
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


@pytest.mark.parametrize("inner_lr", [1e-2])
@pytest.mark.parametrize("outer_lr", [1e-2])
@pytest.mark.parametrize("inner_iters", [2])
@pytest.mark.parametrize("outer_iters", [4])
def test(inner_lr, outer_lr, inner_iters, outer_iters):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    model = Model().to(device)
    slow_params = tuple(model.slow.parameters())
    fast_params = tuple(model.fast.parameters())
    outer_opt = torch.optim.SGD(model.parameters(), lr=outer_lr)

    slow_init = [p.detach().clone() for p in slow_params]
    fast_init = [p.detach().clone() for p in fast_params]

    # Fixed dataset so we can observe clear convergence.
    x = torch.randn(8, 4, device=device)
    y = torch.randint(0, 4, (8,), device=device)

    loss_history = []

    for outer_step in range(outer_iters):
        # Fresh fast-weight clones from the (updated) model each outer step.
        cur_fast = {
            n: p.clone()
            for n, p in model.named_parameters()
            if n.startswith("fast")
        }
        fast_names = list(cur_fast.keys())

        outer_opt.zero_grad()
        inner_losses = []

        # ---- inner loop ----
        for _ in range(inner_iters):
            all_params = dict(model.named_parameters())
            all_params.update(cur_fast)

            logits = functional_call(model, all_params, (x,))
            loss = nn.functional.cross_entropy(logits, y)
            inner_losses.append(loss)

            grads = torch.autograd.grad(
                loss, tuple(cur_fast.values()), create_graph=True,
            )
            cur_fast = {
                n: p - inner_lr * g
                for (n, p), g in zip(cur_fast.items(), grads)
            }

        # ---- outer step ----
        outer_loss = torch.stack(inner_losses).mean()
        outer_loss.backward()
        outer_opt.step()

        loss_val = outer_loss.item()
        loss_history.append(loss_val)
        assert torch.isfinite(torch.tensor(loss_val)), \
            f"outer step {outer_step}: loss is not finite ({loss_val})"

    # ---- convergence checks ----
    k = max(1, outer_iters // 4)
    first_avg = sum(loss_history[:k]) / k
    last_avg = sum(loss_history[-k:]) / k
    assert last_avg < first_avg, (
        f"loss did not decrease: first-quarter avg {first_avg:.4f} "
        f"vs last-quarter avg {last_avg:.4f}"
    )

    # Both parameter groups should have moved from their initial values.
    for init, p in zip(slow_init, slow_params):
        assert not torch.allclose(init, p.detach()), "slow params unchanged after training"
    for init, p in zip(fast_init, fast_params):
        assert not torch.allclose(init, p.detach()), "fast params unchanged after training"
