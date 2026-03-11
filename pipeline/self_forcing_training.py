from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional, Tuple

from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from utils.meta_learning import (
    clone_prime_params,
    build_fast_weights_dict,
    functional_inner_step,
    fast_param_keys,
    fast_params_to_flat,
    fast_params_from_flat,
)
from utils.wan_cache import (
    build_crossattn_cache_entry,
    build_kv_cache_entry,
    clone_flat_cache_state,
    flatten_cache_state,
    unflatten_cache_state,
)
import torch
import torch.distributed as dist


class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length

        self.ttt_enabled = kwargs.pop("ttt_enabled", False)
        self.ttt_inner_lr = kwargs.pop("ttt_inner_lr", 1e-3)
        self.ttt_inner_clip = kwargs.pop("ttt_inner_clip", 1.0)
        self.ttt_first_order = kwargs.pop("ttt_first_order", False)
        self.ttt_rollout_checkpoint_every = max(
            0, int(kwargs.pop("ttt_rollout_checkpoint_every", 0))
        )
        self.disable_model_checkpointing_during_rollout_ckpt = kwargs.pop(
            "disable_model_checkpointing_during_rollout_ckpt", False
        )
        self._last_ttt_loss = torch.tensor(0.0)

    # ------------------------------------------------------------------
    # TTT helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _peel(module):
        """Strip one layer of FSDP / torch.compile wrapping."""
        if hasattr(module, "_fsdp_wrapped_module"):
            module = module._fsdp_wrapped_module
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod
        return module

    def _unwrap_model(self):
        model = self._peel(self.generator.model)
        return model

    @contextmanager
    def _maybe_disable_model_checkpointing(self):
        model = self._unwrap_model()
        previous = getattr(model, "gradient_checkpointing", False)
        if self.disable_model_checkpointing_during_rollout_ckpt and previous:
            model.gradient_checkpointing = False
        try:
            yield
        finally:
            if getattr(model, "gradient_checkpointing", False) != previous:
                model.gradient_checkpointing = previous

    def _capture_cache_metadata(self):
        kv_state = tuple(
            (
                cache["global_end_index"].clone(),
                cache["local_end_index"].clone(),
            )
            for cache in self.kv_cache1
        )
        crossattn_state = tuple(
            cache["is_init"].clone()
            for cache in self.crossattn_cache
        )
        return kv_state, crossattn_state

    def _restore_cache_metadata(self, cache_state) -> None:
        kv_state, crossattn_state = cache_state
        for cache, (global_end, local_end) in zip(self.kv_cache1, kv_state):
            cache["global_end_index"].copy_(global_end)
            cache["local_end_index"].copy_(local_end)
        for cache, is_init in zip(self.crossattn_cache, crossattn_state):
            cache["is_init"] = is_init.clone()

    def _validate_rollout_checkpointing(
        self,
        num_output_frames: int,
    ) -> None:
        max_cached_frames = self.kv_cache_size // self.frame_seq_length
        if num_output_frames > max_cached_frames:
            raise ValueError(
                "Rollout checkpointing assumes the KV cache only appends during "
                "the rollout. Increase num_max_frames or disable "
                "ttt_rollout_checkpoint_every."
            )

    def _rollout_checkpoint_contexts(self):
        if self.disable_model_checkpointing_during_rollout_ckpt:
            # Escape hatch for the historical FSDP replay mismatch. The
            # explicit-cache rollout path now keeps nested chunk+layer
            # checkpointing enabled by default.
            return (
                self._maybe_disable_model_checkpointing(),
                self._maybe_disable_model_checkpointing(),
            )
        return nullcontext(), nullcontext()

    def _run_decoding_blocks(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        kv_cache: List[dict],
        crossattn_cache: List[dict],
        all_num_frames: List[int],
        exit_flags: List[int],
        num_input_frames: int,
        start_gradient_frame_index: int,
        current_start_frame: int,
        block_start: int,
        block_end: int,
        fast_values: Optional[Tuple[torch.Tensor, ...]] = None,
        fast_keys: Optional[List[Tuple[int, str]]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        fast_params = None
        if fast_keys is not None and fast_values is not None:
            fast_params = fast_params_from_flat(fast_keys, fast_values)

        outputs: List[torch.Tensor] = []
        inner_losses: List[torch.Tensor] = []
        batch_size = noise.shape[0]

        for block_index in range(block_start, block_end):
            current_num_frames = all_num_frames[block_index]
            noisy_input = noise[
                :,
                current_start_frame - num_input_frames:
                current_start_frame + current_num_frames - num_input_frames,
            ]

            if self.same_step_across_blocks:
                exit_idx = exit_flags[0]
            else:
                exit_idx = exit_flags[block_index]

            fw = build_fast_weights_dict(fast_params) if fast_params else None

            for index, current_timestep in enumerate(self.denoising_step_list):
                is_exit = index == exit_idx
                timestep = torch.full(
                    [batch_size, current_num_frames],
                    current_timestep,
                    device=noise.device,
                    dtype=torch.int64,
                )

                if not is_exit:
                    with torch.no_grad():
                        _, denoised_pred, kv_cache, crossattn_cache = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=kv_cache,
                            crossattn_cache=crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            fast_weights=fw,
                            return_cache=True,
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames],
                                device=noise.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    use_grad = self.ttt_enabled or (
                        current_start_frame >= start_gradient_frame_index
                    )
                    with torch.set_grad_enabled(use_grad):
                        _, denoised_pred, kv_cache, crossattn_cache = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=kv_cache,
                            crossattn_cache=crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            fast_weights=fw,
                            return_cache=True,
                        )
                    break

            outputs.append(denoised_pred)

            if fast_params is not None:
                inner_loss = self._compute_inner_loss(
                    denoised_pred.detach(),
                    fw,
                    conditional_dict,
                    current_start_frame,
                    kv_cache,
                    crossattn_cache,
                )
                inner_losses.append(inner_loss.detach())
                fast_params = functional_inner_step(
                    fast_params,
                    inner_loss,
                    self.ttt_inner_lr,
                    self.ttt_inner_clip,
                    create_graph=not self.ttt_first_order,
                )
                del inner_loss
                torch.cuda.empty_cache()

            context_timestep = torch.ones_like(timestep) * self.context_noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames],
                    device=noise.device,
                    dtype=torch.long,
                ),
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                _, _, kv_cache, crossattn_cache = self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    return_cache=True,
                )

            current_start_frame += current_num_frames

        result: List[torch.Tensor] = []
        if fast_keys is not None and fast_params is not None:
            result.extend(fast_params_to_flat(fast_params, fast_keys))
        result.extend(flatten_cache_state(kv_cache, crossattn_cache))
        result.extend(outputs)
        result.extend(inner_losses)
        return tuple(result)

    @torch.enable_grad()
    def _compute_inner_loss(
        self,
        pseudo_gt: torch.Tensor,
        fast_weights: Dict[int, Dict[str, torch.Tensor]],
        conditional_dict: dict,
        current_start_frame: int,
        kv_cache: List[dict],
        crossattn_cache: List[dict],
    ) -> torch.Tensor:
        """Flow-matching denoising loss for the TTT inner loop.

        Runs the generator under ``torch.no_grad()`` with
        ``detach_prime_input=True``.  The slow path (attention/FFN)
        runs without registering FSDP backward hooks.  PrimeFFN blocks
        and the head run under local ``torch.enable_grad()`` so the
        output carries grad through fast weights only.  The resulting
        ``autograd.grad`` touches PrimeFFN (FSDP-ignored) and passes
        through the head (a single non-checkpointed FSDP unit) without
        corrupting the 30-block execution-order state.
        """
        batch_size, num_frames = pseudo_gt.shape[:2]
        device = pseudo_gt.device

        with torch.no_grad():
            step_indices = torch.randint(
                0, len(self.denoising_step_list),
                [batch_size, num_frames], device=device,
            )
            ttt_timestep = self.denoising_step_list.to(device)[step_indices]
            ttt_noise = torch.randn_like(pseudo_gt)
            noisy_input = self.scheduler.add_noise(
                pseudo_gt.flatten(0, 1),
                ttt_noise.flatten(0, 1),
                ttt_timestep.flatten(0, 1),
            ).unflatten(0, (batch_size, num_frames))

            flow_pred, _, _, _ = self.generator(
                noisy_image_or_video=noisy_input.detach(),
                conditional_dict={k: v.detach() if torch.is_tensor(v) else v
                                  for k, v in conditional_dict.items()},
                timestep=ttt_timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                update_cache=False,
                fast_weights=fast_weights,
                detach_prime_input=True,
                return_cache=True,
            )

        target = ttt_noise - pseudo_gt
        return torch.mean((flow_pred.float() - target.float()) ** 2)

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        kv_cache = self.kv_cache1
        crossattn_cache = self.crossattn_cache
        # if self.kv_cache1 is None:
        #     self._initialize_kv_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device,
        #     )
        #     self._initialize_crossattn_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device
        #     )
        # else:
        #     # reset cross attn cache
        #     for block_index in range(self.num_transformer_blocks):
        #         self.crossattn_cache[block_index]["is_init"] = False
        #     # reset kv cache
        #     for block_index in range(len(self.kv_cache1)):
        #         self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)
        #         self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                _, _, kv_cache, crossattn_cache = self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    return_cache=True,
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(
            len(all_num_frames), num_denoising_steps, noise.device)

        # --- TTT: clone fast weights (None when ttt_enabled=False) ---
        fast_params = None
        inner_losses: List[torch.Tensor] = []
        if self.ttt_enabled:
            fast_params = clone_prime_params(self._unwrap_model())

        # For non-TTT, only the last 21 frames get gradient
        start_gradient_frame_index = num_output_frames - 21

        rollout_checkpoint = (
            fast_params is not None and self.ttt_rollout_checkpoint_every > 0
        )
        fast_keys = fast_param_keys(fast_params) if fast_params else None
        fast_values = (
            fast_params_to_flat(fast_params, fast_keys)
            if fast_params is not None and fast_keys is not None
            else None
        )

        if rollout_checkpoint:
            self._validate_rollout_checkpointing(num_output_frames)
            block_start = 0
            while block_start < len(all_num_frames):
                block_end = min(
                    block_start + self.ttt_rollout_checkpoint_every,
                    len(all_num_frames),
                )
                chunk_num_blocks = block_end - block_start
                chunk_start_frame = current_start_frame
                flat_cache = flatten_cache_state(kv_cache, crossattn_cache)

                def run_chunk(
                    flat_fast_values,
                    *flat_cache_values,
                    chunk_start_frame=chunk_start_frame,
                    block_start=block_start,
                    block_end=block_end,
                ):
                    flat_cache_values = clone_flat_cache_state(flat_cache_values)
                    chunk_kv_cache, chunk_crossattn_cache = unflatten_cache_state(
                        flat_cache_values,
                        num_kv_blocks=self.num_transformer_blocks,
                        num_crossattn_blocks=self.num_transformer_blocks,
                    )
                    return self._run_decoding_blocks(
                        noise=noise,
                        conditional_dict=conditional_dict,
                        kv_cache=chunk_kv_cache,
                        crossattn_cache=chunk_crossattn_cache,
                        all_num_frames=all_num_frames,
                        exit_flags=exit_flags,
                        num_input_frames=num_input_frames,
                        start_gradient_frame_index=start_gradient_frame_index,
                        current_start_frame=chunk_start_frame,
                        block_start=block_start,
                        block_end=block_end,
                        fast_values=flat_fast_values,
                        fast_keys=fast_keys,
                    )

                with torch.utils.checkpoint.set_checkpoint_early_stop(False):
                    result = torch.utils.checkpoint.checkpoint(
                        run_chunk,
                        fast_values,
                        *flat_cache,
                        context_fn=self._rollout_checkpoint_contexts,
                        use_reentrant=False,
                    )
                num_fast = len(fast_values)
                num_cache = len(flat_cache)
                fast_values = result[:num_fast]
                kv_cache, crossattn_cache = unflatten_cache_state(
                    result[num_fast:num_fast + num_cache],
                    num_kv_blocks=self.num_transformer_blocks,
                    num_crossattn_blocks=self.num_transformer_blocks,
                )
                chunk_outputs = result[num_fast + num_cache:num_fast + num_cache + chunk_num_blocks]
                chunk_inner_losses = result[num_fast + num_cache + chunk_num_blocks:]

                for local_index, denoised_pred in enumerate(chunk_outputs):
                    current_num_frames = all_num_frames[block_start + local_index]
                    output[
                        :,
                        current_start_frame:current_start_frame + current_num_frames,
                    ] = denoised_pred
                    current_start_frame += current_num_frames
                inner_losses.extend(chunk_inner_losses)
                block_start = block_end

            fast_params = fast_params_from_flat(fast_keys, fast_values)
        else:
            result = self._run_decoding_blocks(
                noise=noise,
                conditional_dict=conditional_dict,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                all_num_frames=all_num_frames,
                exit_flags=exit_flags,
                num_input_frames=num_input_frames,
                start_gradient_frame_index=start_gradient_frame_index,
                current_start_frame=current_start_frame,
                block_start=0,
                block_end=len(all_num_frames),
                fast_values=fast_values,
                fast_keys=fast_keys,
            )
            num_fast = len(fast_values) if fast_values is not None else 0
            num_cache = len(flatten_cache_state(kv_cache, crossattn_cache))
            kv_cache, crossattn_cache = unflatten_cache_state(
                result[num_fast:num_fast + num_cache],
                num_kv_blocks=self.num_transformer_blocks,
                num_crossattn_blocks=self.num_transformer_blocks,
            )
            block_outputs = result[num_fast + num_cache:num_fast + num_cache + len(all_num_frames)]
            block_inner_losses = result[num_fast + num_cache + len(all_num_frames):]

            for block_index, denoised_pred in enumerate(block_outputs):
                current_num_frames = all_num_frames[block_index]
                output[
                    :,
                    current_start_frame:current_start_frame + current_num_frames,
                ] = denoised_pred
                current_start_frame += current_num_frames
            inner_losses.extend(block_inner_losses)
            if fast_keys is not None and fast_values is not None:
                fast_params = fast_params_from_flat(
                    fast_keys,
                    result[:num_fast],
                )

        self.kv_cache1 = kv_cache
        self.crossattn_cache = crossattn_cache

        # bookkeeping
        if inner_losses:
            self._last_ttt_loss = torch.stack(inner_losses).mean()

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                build_kv_cache_entry(
                    torch.zeros(
                        [batch_size, self.kv_cache_size, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    torch.zeros(
                        [batch_size, self.kv_cache_size, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    torch.tensor([0], dtype=torch.long, device=device),
                    torch.tensor([0], dtype=torch.long, device=device),
                )
            )

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                build_crossattn_cache_entry(
                    torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    torch.tensor([False], dtype=torch.bool, device=device),
                )
            )
        self.crossattn_cache = crossattn_cache
