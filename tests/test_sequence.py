"""Meta-learning: sequence-chunked inner loop."""
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
@pytest.mark.parametrize("num_chunks", [2])
@pytest.mark.parametrize("chunk_len", [2])
def test(inner_lr, outer_lr, num_chunks, chunk_len):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    model = Model().to(device)
    slow_params = tuple(model.slow.parameters())
    fast_params = tuple(model.fast.parameters())
    outer_opt = torch.optim.SGD(model.parameters(), lr=outer_lr)

    # Long sequence split into chunks
    seq_len = num_chunks * chunk_len
    full_x = torch.randn(seq_len, 4, device=device)
    full_y = torch.randint(0, 4, (seq_len,), device=device)

    x_chunks = full_x.split(chunk_len)
    y_chunks = full_y.split(chunk_len)

    cur_fast = {
        n: p.clone()
        for n, p in model.named_parameters()
        if n.startswith("fast")
    }

    outer_opt.zero_grad()
    chunk_losses = []

    # Walk through the sequence: each chunk = one inner step.
    for x_chunk, y_chunk in zip(x_chunks, y_chunks):
        all_params = dict(model.named_parameters())
        all_params.update(cur_fast)

        logits = functional_call(model, all_params, (x_chunk,))
        loss = nn.functional.cross_entropy(logits, y_chunk)
        chunk_losses.append(loss)

        grads = torch.autograd.grad(
            loss, tuple(cur_fast.values()), create_graph=True,
        )
        cur_fast = {
            n: p - inner_lr * g
            for (n, p), g in zip(cur_fast.items(), grads)
        }

    # Outer loss: mean over all chunks
    outer_loss = torch.stack(chunk_losses).mean()

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

    # Higher-order: outer grads should back-prop into the initial fast weights.
    higher_order_loss = sum(g.pow(2).sum() for g in outer_grads)
    higher_order_grads = torch.autograd.grad(
        higher_order_loss, fast_params, retain_graph=True,
    )

    for grad in higher_order_grads:
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum().item() > 0

    # Outer optimizer step should change both slow and fast params.
    outer_opt.zero_grad()
    slow_before = [p.detach().clone() for p in slow_params]
    fast_before = [p.detach().clone() for p in fast_params]
    outer_loss.backward()
    outer_opt.step()

    for before, after in zip(slow_before, slow_params):
        assert not torch.allclose(before, after.detach())
    for before, after in zip(fast_before, fast_params):
        assert not torch.allclose(before, after.detach())
