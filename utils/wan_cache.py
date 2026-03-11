from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


KV_CACHE_TENSORS_PER_BLOCK = 4
CROSSATTN_CACHE_TENSORS_PER_BLOCK = 3


def ensure_bool_tensor(value, *, device=None) -> torch.Tensor:
    if torch.is_tensor(value):
        if device is None or value.device == torch.device(device):
            return value.to(dtype=torch.bool)
        return value.to(device=device, dtype=torch.bool)
    return torch.tensor([bool(value)], device=device, dtype=torch.bool)


def build_kv_cache_entry(
    k: torch.Tensor,
    v: torch.Tensor,
    global_end_index: torch.Tensor,
    local_end_index: torch.Tensor,
) -> dict:
    return {
        "k": k,
        "v": v,
        "global_end_index": global_end_index,
        "local_end_index": local_end_index,
    }


def build_crossattn_cache_entry(
    k: torch.Tensor,
    v: torch.Tensor,
    is_init,
) -> dict:
    return {
        "k": k,
        "v": v,
        "is_init": ensure_bool_tensor(is_init, device=k.device),
    }


def flatten_cache_state(
    kv_cache: Sequence[dict],
    crossattn_cache: Sequence[dict],
) -> Tuple[torch.Tensor, ...]:
    flat_cache: List[torch.Tensor] = []
    for cache in kv_cache:
        flat_cache.extend(
            [
                cache["k"],
                cache["v"],
                cache["global_end_index"],
                cache["local_end_index"],
            ]
        )
    for cache in crossattn_cache:
        flat_cache.extend(
            [
                cache["k"],
                cache["v"],
                ensure_bool_tensor(cache["is_init"], device=cache["k"].device),
            ]
        )
    return tuple(flat_cache)


def unflatten_cache_state(
    flat_cache: Sequence[torch.Tensor],
    *,
    num_kv_blocks: int,
    num_crossattn_blocks: int,
) -> tuple[list[dict], list[dict]]:
    flat_index = 0
    kv_cache: list[dict] = []
    crossattn_cache: list[dict] = []

    for _ in range(num_kv_blocks):
        kv_cache.append(
            build_kv_cache_entry(
                flat_cache[flat_index],
                flat_cache[flat_index + 1],
                flat_cache[flat_index + 2],
                flat_cache[flat_index + 3],
            )
        )
        flat_index += KV_CACHE_TENSORS_PER_BLOCK

    for _ in range(num_crossattn_blocks):
        crossattn_cache.append(
            build_crossattn_cache_entry(
                flat_cache[flat_index],
                flat_cache[flat_index + 1],
                flat_cache[flat_index + 2],
            )
        )
        flat_index += CROSSATTN_CACHE_TENSORS_PER_BLOCK

    if flat_index != len(flat_cache):
        raise ValueError(
            f"Unexpected flattened cache length {len(flat_cache)} "
            f"for {num_kv_blocks} kv blocks and {num_crossattn_blocks} cross-attn blocks."
        )
    return kv_cache, crossattn_cache


def clone_flat_cache_state(
    flat_cache: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.clone() for tensor in flat_cache)


def overwrite_cache_state_(
    dst_kv_cache: list[dict] | None,
    dst_crossattn_cache: list[dict] | None,
    src_kv_cache: Sequence[dict],
    src_crossattn_cache: Sequence[dict],
) -> None:
    if dst_kv_cache is not None:
        dst_kv_cache[:] = list(src_kv_cache)
    if dst_crossattn_cache is not None:
        dst_crossattn_cache[:] = list(src_crossattn_cache)
