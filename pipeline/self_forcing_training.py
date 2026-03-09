from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from utils.meta_learning import (
    clone_prime_params,
    build_fast_weights_dict,
    functional_inner_step,
)
from typing import Dict, List, Optional, Tuple
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

    @torch.enable_grad()
    def _compute_inner_loss(
        self,
        pseudo_gt: torch.Tensor,
        fast_weights: Dict[int, Dict[str, torch.Tensor]],
        conditional_dict: dict,
        current_start_frame: int,
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

            flow_pred, _ = self.generator(
                noisy_image_or_video=noisy_input.detach(),
                conditional_dict={k: v.detach() if torch.is_tensor(v) else v
                                  for k, v in conditional_dict.items()},
                timestep=ttt_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                update_cache=False,
                fast_weights=fast_weights,
                detach_prime_input=True,
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
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
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

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
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

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                is_exit = (index == exit_idx)
                timestep = torch.full(
                    [batch_size, current_num_frames],
                    current_timestep,
                    device=noise.device, dtype=torch.int64,
                )

                if not is_exit:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            fast_weights=fw,
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames],
                                device=noise.device, dtype=torch.long),
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # TTT: always run exit step with grad
                    # Non-TTT: only last 21 frames get grad
                    use_grad = self.ttt_enabled or (
                        current_start_frame >= start_gradient_frame_index)
                    with torch.set_grad_enabled(use_grad):
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            fast_weights=fw,
                        )
                    break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.2b: TTT inner-loop update.
            # The inner-loop forward runs under no_grad through the FSDP
            # wrapper with detach_prime_input=True.  Only PrimeFFN outputs
            # carry grad (through fast weights, which are FSDP-ignored),
            # so autograd.grad never triggers FSDP backward hooks.
            if fast_params is not None:
                inner_loss = self._compute_inner_loss(
                    denoised_pred.detach(), fw,
                    conditional_dict, current_start_frame,
                )
                inner_losses.append(inner_loss.detach())
                fast_params = functional_inner_step(
                    fast_params, inner_loss,
                    self.ttt_inner_lr, self.ttt_inner_clip,
                    create_graph=not self.ttt_first_order)
                del inner_loss
                torch.cuda.empty_cache()

            # Step 3.3: rerun with timestep zero to update the cache
            new_fw = build_fast_weights_dict(fast_params) if fast_params else None
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

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
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
