"""Pure-function utilities for MAML-style meta-learning with fast weights.

All helpers are stateless and operate on the canonical fast-params
representation: ``Dict[int, Dict[str, Tensor]]`` mapping
``block_idx → {param_name: tensor}``.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# ------------------------------------------------------------------
# Clone / build
# ------------------------------------------------------------------

@torch.enable_grad()
def clone_prime_params(
    model: nn.Module,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Clone PrimeFFN parameters per suffix block.

    Returns ``{block_idx: {param_name: cloned_tensor}}``.
    Decorated with ``@torch.enable_grad()`` so clones keep
    ``requires_grad=True`` even when called under ``no_grad()``
    (e.g. during the critic step).
    """
    fast: Dict[int, Dict[str, torch.Tensor]] = {}
    for idx, blk in enumerate(model.blocks):
        pfn = blk.prime_ffn if hasattr(blk, "prime_ffn") else None
        if hasattr(blk, "_fsdp_wrapped_module"):
            inner = blk._fsdp_wrapped_module
            if hasattr(inner, "prime_ffn"):
                pfn = inner.prime_ffn
        if pfn is not None:
            fast[idx] = {
                name: p.clone()
                for name, p in pfn.named_parameters()
            }
    return fast


def build_fast_weights_dict(
    fast_params: Dict[int, Dict[str, torch.Tensor]],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Trivial pass-through; exists as an extension point for vmap."""
    return fast_params


# ------------------------------------------------------------------
# Flat-tuple conversions (needed by torch.utils.checkpoint)
# ------------------------------------------------------------------

def fast_param_keys(
    fast_params: Dict[int, Dict[str, torch.Tensor]],
) -> List[Tuple[int, str]]:
    keys: List[Tuple[int, str]] = []
    for blk in sorted(fast_params):
        for name in sorted(fast_params[blk]):
            keys.append((blk, name))
    return keys


def fast_params_to_flat(
    fast_params: Dict[int, Dict[str, torch.Tensor]],
    keys: List[Tuple[int, str]],
) -> Tuple[torch.Tensor, ...]:
    return tuple(fast_params[b][n] for b, n in keys)


def fast_params_from_flat(
    keys: List[Tuple[int, str]],
    values: Tuple[torch.Tensor, ...],
) -> Dict[int, Dict[str, torch.Tensor]]:
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for (blk, name), val in zip(keys, values):
        if blk not in out:
            out[blk] = {}
        out[blk][name] = val
    return out


# ------------------------------------------------------------------
# Functional inner-loop optimiser
# ------------------------------------------------------------------

@torch.enable_grad()
def functional_inner_step(
    fast_params: Dict[int, Dict[str, torch.Tensor]],
    inner_loss: torch.Tensor,
    lr: float,
    max_grad_norm: float = 0.0,
    create_graph: bool = True,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """One functional SGD update on all fast weights.

    Args:
        create_graph: Keep the graph for second-order gradients (needed
            during training so the outer loss can differentiate through
            the inner update chain).  Set to ``False`` at inference.
    """
    all_tensors: List[torch.Tensor] = []
    keys: List[Tuple[int, str]] = []
    for blk_idx in sorted(fast_params):
        for name in sorted(fast_params[blk_idx]):
            keys.append((blk_idx, name))
            all_tensors.append(fast_params[blk_idx][name])

    grads = torch.autograd.grad(
        inner_loss, all_tensors, create_graph=create_graph,
        allow_unused=True,
    )
    grads = [g if g is not None else torch.zeros_like(t)
             for g, t in zip(grads, all_tensors)]

    if max_grad_norm > 0:
        total_norm = torch.sqrt(
            sum(g.float().pow(2).sum() for g in grads)
        )
        clip_coef = torch.clamp(
            max_grad_norm / (total_norm + 1e-6), max=1.0,
        )
    else:
        clip_coef = 1.0

    new_fast: Dict[int, Dict[str, torch.Tensor]] = {}
    for (blk_idx, name), grad in zip(keys, grads):
        if blk_idx not in new_fast:
            new_fast[blk_idx] = {}
        new_fast[blk_idx][name] = (
            fast_params[blk_idx][name] - lr * clip_coef * grad
        )
    return new_fast
