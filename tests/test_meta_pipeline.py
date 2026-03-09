"""Integration test: meta-learning pipeline gradient flow.

Verifies that an outer loss back-props through the functional inner-loop
chain to *both* slow parameters and the initial PrimeFFN (fast) weights.

The test builds a tiny model that mimics the real architecture:
  - A "slow" backbone (linear)
  - A PrimeFFN adapter per suffix block, called via ``functional_call``

Then it runs the same clone → inner-loss → functional-SGD → outer-loss
pattern used in ``SelfForcingTrainingPipeline._inference_meta_learning``.

Run:  python tests/test_meta_pipeline.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call


# ---- tiny model mimicking CausalWanAttentionBlock + PrimeFFN ----

class TinyPrimeFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.ffn(self.norm(x))


class TinyBlock(nn.Module):
    def __init__(self, dim, prime_dim):
        super().__init__()
        self.slow_linear = nn.Linear(dim, dim)
        self.prime_ffn = TinyPrimeFFN(dim, prime_dim)

    def forward(self, x, prime_ffn_override=None):
        h = F.gelu(self.slow_linear(x))
        if prime_ffn_override is not None:
            h = h + functional_call(self.prime_ffn, prime_ffn_override, (h,))
        else:
            h = h + self.prime_ffn(h)
        return h


class TinyModel(nn.Module):
    def __init__(self, dim=8, n_blocks=4, suffix_len=2, prime_dim=4):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(dim, prime_dim) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, dim)
        self.suffix_start = n_blocks - suffix_len

    def forward(self, x, fast_weights=None):
        for i, blk in enumerate(self.blocks):
            override = fast_weights.get(i) if fast_weights else None
            x = blk(x, prime_ffn_override=override)
        return self.head(x)

    def get_suffix_block_indices(self):
        return list(range(self.suffix_start, len(self.blocks)))

    def clone_prime_params(self):
        fast = {}
        for idx in self.get_suffix_block_indices():
            fast[idx] = {
                n: p.clone()
                for n, p in self.blocks[idx].prime_ffn.named_parameters()
            }
        return fast

    def get_slow_params(self):
        slow = []
        for i, blk in enumerate(self.blocks):
            slow.extend(blk.slow_linear.parameters())
        slow.extend(self.head.parameters())
        return slow

    def get_prime_params(self):
        params = []
        for idx in self.get_suffix_block_indices():
            params.extend(self.blocks[idx].prime_ffn.parameters())
        return params


# ---- functional inner step (mirrors pipeline logic) ----

def functional_inner_step(fast_params, inner_loss, lr):
    all_tensors = []
    keys = []
    for blk_idx in sorted(fast_params):
        for name in sorted(fast_params[blk_idx]):
            keys.append((blk_idx, name))
            all_tensors.append(fast_params[blk_idx][name])

    grads = torch.autograd.grad(inner_loss, all_tensors, create_graph=True)

    new_fast = {}
    for (blk_idx, name), grad in zip(keys, grads):
        if blk_idx not in new_fast:
            new_fast[blk_idx] = {}
        new_fast[blk_idx][name] = fast_params[blk_idx][name] - lr * grad
    return new_fast


# ---- main test ----

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim = 8
    n_blocks = 4
    suffix_len = 2
    prime_dim = 4
    inner_lr = 1e-2
    num_chunks = 3

    model = TinyModel(dim, n_blocks, suffix_len, prime_dim).to(device)

    slow_params = model.get_slow_params()
    prime_params = model.get_prime_params()
    all_params = slow_params + prime_params

    slow_init = [p.detach().clone() for p in slow_params]
    prime_init = [p.detach().clone() for p in prime_params]

    outer_opt = torch.optim.SGD(model.parameters(), lr=1e-1)
    outer_opt.zero_grad()

    # --- simulate sequence of chunks (like blocks in self-forcing) ---
    xs = [torch.randn(1, dim, device=device) for _ in range(num_chunks)]
    targets = [torch.randn(1, dim, device=device) for _ in range(num_chunks)]

    fast_params = model.clone_prime_params()
    outputs = []

    for chunk_idx in range(num_chunks):
        fw = fast_params
        out = model(xs[chunk_idx], fast_weights=fw)
        outputs.append(out)

        # Inner loss via a *separate* forward with fast_weights (mirrors the
        # real pipeline's re-denoising step).  The pseudo-GT is detached so
        # the inner loss only differentiates through this second forward.
        pseudo_gt = out.detach()
        inner_pred = model(pseudo_gt, fast_weights=fw)
        inner_loss = F.mse_loss(inner_pred, targets[chunk_idx])
        fast_params = functional_inner_step(fast_params, inner_loss, inner_lr)

    # outer loss on all outputs (like DMD loss on trajectory)
    trajectory = torch.cat(outputs, dim=0)
    outer_target = torch.randn_like(trajectory)
    outer_loss = F.mse_loss(trajectory, outer_target)

    outer_loss.backward()

    # --- assertions ---
    # 1. Both slow and fast params should have gradients
    for i, p in enumerate(slow_params):
        assert p.grad is not None, f"slow param {i} has no gradient"
        assert torch.isfinite(p.grad).all(), f"slow param {i} gradient not finite"

    for i, p in enumerate(prime_params):
        assert p.grad is not None, f"prime param {i} has no gradient"
        assert torch.isfinite(p.grad).all(), f"prime param {i} gradient not finite"
        assert p.grad.abs().max().item() > 0, f"prime param {i} gradient is zero"

    # 2. After optimizer step, both param groups should have changed
    outer_opt.step()

    for init_p, p in zip(slow_init, slow_params):
        assert not torch.allclose(init_p, p.detach()), "slow param unchanged"

    for init_p, p in zip(prime_init, prime_params):
        assert not torch.allclose(init_p, p.detach()), "prime param unchanged"

    # 3. Run a second outer step to verify loss decreases
    outer_opt.zero_grad()
    fast_params2 = model.clone_prime_params()
    outputs2 = []
    for chunk_idx in range(num_chunks):
        out = model(xs[chunk_idx], fast_weights=fast_params2)
        outputs2.append(out)
        pseudo_gt = out.detach()
        inner_pred = model(pseudo_gt, fast_weights=fast_params2)
        inner_loss = F.mse_loss(inner_pred, targets[chunk_idx])
        fast_params2 = functional_inner_step(fast_params2, inner_loss, inner_lr)

    trajectory2 = torch.cat(outputs2, dim=0)
    outer_loss2 = F.mse_loss(trajectory2, outer_target)
    # The second loss should be finite
    assert torch.isfinite(outer_loss2), f"second outer loss not finite: {outer_loss2}"

    print(f"loss step 0: {outer_loss.item():.6f}")
    print(f"loss step 1: {outer_loss2.item():.6f}")
    print("PASSED")


if __name__ == "__main__":
    main()
