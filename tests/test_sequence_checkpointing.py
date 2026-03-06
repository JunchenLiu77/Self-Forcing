"""Meta-learning: sequence-chunked inner loop with gradient checkpointing.

Chunks are grouped into remat segments wrapped in torch.utils.checkpoint.
The test verifies checkpointed and plain forward/backward produce identical results.
"""
import torch
import torch.nn as nn
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint as ckpt
import pytest


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.slow = nn.Linear(4, 8)
        self.fast = nn.Linear(8, 4)

    def forward(self, x):
        return self.fast(self.slow(x))


# ---- shared single-chunk step ----

def _inner_step(model, cur_fast, x_chunk, y_chunk, inner_lr):
    all_params = dict(model.named_parameters())
    all_params.update(cur_fast)
    logits = functional_call(model, all_params, (x_chunk,))
    loss = nn.functional.cross_entropy(logits, y_chunk)
    grads = torch.autograd.grad(
        loss, tuple(cur_fast.values()), create_graph=True,
    )
    new_fast = {
        n: p - inner_lr * g
        for (n, p), g in zip(cur_fast.items(), grads)
    }
    return new_fast, loss


# ---- plain (no checkpointing) ----

def run_plain(model, x_chunks, y_chunks, init_fast, inner_lr):
    cur_fast = {n: v.clone() for n, v in init_fast.items()}
    losses = []
    for x_c, y_c in zip(x_chunks, y_chunks):
        cur_fast, loss = _inner_step(model, cur_fast, x_c, y_c, inner_lr)
        losses.append(loss)
    return torch.stack(losses)


# ---- checkpointed ----

def _segment_body(fast_values, x_seg, y_seg, model, fast_names, inner_lr):
    cur_fast = dict(zip(fast_names, fast_values))
    losses = []
    for i in range(x_seg.shape[0]):
        cur_fast, loss = _inner_step(
            model, cur_fast, x_seg[i], y_seg[i], inner_lr,
        )
        losses.append(loss)
    # Return flat tuple: (updated_fast_0, updated_fast_1, ..., stacked_losses)
    return tuple(cur_fast.values()) + (torch.stack(losses),)


def run_checkpointed(model, x_chunks, y_chunks, init_fast, inner_lr, remat_n):
    fast_names = list(init_fast.keys())
    num_fast = len(fast_names)
    fast_values = tuple(v.clone() for v in init_fast.values())

    num_chunks = len(x_chunks)
    assert num_chunks % remat_n == 0, (
        f"num_chunks ({num_chunks}) must be divisible by remat_n ({remat_n})"
    )
    n_segments = num_chunks // remat_n

    x_stacked = torch.stack(list(x_chunks))
    y_stacked = torch.stack(list(y_chunks))
    x_segments = x_stacked.reshape(n_segments, remat_n, *x_stacked.shape[1:])
    y_segments = y_stacked.reshape(n_segments, remat_n, *y_stacked.shape[1:])

    all_losses = []
    for seg_idx in range(n_segments):
        result = ckpt(
            _segment_body,
            fast_values,
            x_segments[seg_idx],
            y_segments[seg_idx],
            model,
            fast_names,
            inner_lr,
            use_reentrant=False,
        )
        fast_values = result[:num_fast]
        all_losses.append(result[num_fast])

    return torch.cat(all_losses)


# ---- test ----

@pytest.mark.parametrize("inner_lr", [1e-3])
@pytest.mark.parametrize("outer_lr", [1e-3])
@pytest.mark.parametrize("num_chunks", [4])
@pytest.mark.parametrize("chunk_len", [2])
@pytest.mark.parametrize("remat_n", [1, 2, 4])
def test_checkpoint_equivalence(
    inner_lr, outer_lr, num_chunks, chunk_len, remat_n,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    model = Model().to(device)
    slow_params = tuple(model.slow.parameters())
    fast_params = tuple(model.fast.parameters())

    seq_len = num_chunks * chunk_len
    full_x = torch.randn(seq_len, 4, device=device)
    full_y = torch.randint(0, 4, (seq_len,), device=device)
    x_chunks = full_x.split(chunk_len)
    y_chunks = full_y.split(chunk_len)

    init_fast = {
        n: p.clone()
        for n, p in model.named_parameters()
        if n.startswith("fast")
    }

    # ---- run both versions ----
    losses_plain = run_plain(model, x_chunks, y_chunks, init_fast, inner_lr)
    losses_ckpt = run_checkpointed(
        model, x_chunks, y_chunks, init_fast, inner_lr, remat_n,
    )

    outer_loss_plain = losses_plain.mean()
    outer_loss_ckpt = losses_ckpt.mean()

    # Forward losses must match exactly.
    assert torch.allclose(losses_plain, losses_ckpt), (
        f"loss mismatch: max diff = {(losses_plain - losses_ckpt).abs().max()}"
    )

    # Outer gradients must match.
    all_params = slow_params + fast_params
    grads_plain = torch.autograd.grad(
        outer_loss_plain, all_params, retain_graph=False,
    )
    grads_ckpt = torch.autograd.grad(
        outer_loss_ckpt, all_params, retain_graph=True,
    )

    for g1, g2 in zip(grads_plain, grads_ckpt):
        assert torch.allclose(g1, g2, atol=1e-6), (
            f"grad mismatch: max diff = {(g1 - g2).abs().max()}"
        )

    # Sanity: checkpointed gradients are finite and non-zero.
    for g in grads_ckpt:
        assert torch.isfinite(g).all()
        assert g.abs().max().item() > 0

    # Optimizer step should change both param groups.
    outer_opt = torch.optim.SGD(model.parameters(), lr=outer_lr)
    outer_opt.zero_grad()
    slow_before = [p.detach().clone() for p in slow_params]
    fast_before = [p.detach().clone() for p in fast_params]
    outer_loss_ckpt.backward()
    outer_opt.step()

    for before, after in zip(slow_before, slow_params):
        assert not torch.allclose(before, after.detach())
    for before, after in zip(fast_before, fast_params):
        assert not torch.allclose(before, after.detach())
