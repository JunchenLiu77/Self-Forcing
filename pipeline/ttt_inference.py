"""TTT-naive inference pipeline.

Extends CausalInferencePipeline with per-block test-time training:
after generating each frame block, compute a self-supervised diffusion
(flow-prediction) loss on the generated frames and take an AdamW step
on the PrimeFFN adapter parameters.
"""

from typing import List, Optional

import torch

from pipeline.causal_inference import CausalInferencePipeline
from utils.wan_wrapper import WanDiffusionWrapper


class TTTInferencePipeline(CausalInferencePipeline):
    """CausalInferencePipeline + per-block TTT weight updates."""

    def __init__(
        self,
        args,
        device,
        generator=None,
        text_encoder=None,
        vae=None,
    ):
        super().__init__(args, device, generator, text_encoder, vae)
        self.ttt_inner_lr = getattr(args, "ttt_inner_lr", 1e-3)
        self.ttt_inner_clip = getattr(args, "ttt_inner_clip", 1.0)
        self.ttt_min_timestep = getattr(args, "ttt_min_timestep", 20)
        self.ttt_max_timestep = getattr(args, "ttt_max_timestep", 980)
        self.ttt_inner_b1 = getattr(args, "ttt_inner_b1", 0.9)
        self.ttt_inner_b2 = getattr(args, "ttt_inner_b2", 0.999)
        self._inner_optimizer = None

    # ------------------------------------------------------------------
    # TTT helpers
    # ------------------------------------------------------------------
    def _unwrap_model(self):
        model = self.generator.model
        if hasattr(model, "_fsdp_wrapped_module"):
            model = model._fsdp_wrapped_module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    def _get_prime_params(self):
        return self._unwrap_model().get_prime_parameter_list()

    def _reset_prime_and_optimizer(self):
        """Zero-reinitialise prime adapters and create a fresh AdamW optimizer."""
        self._unwrap_model().reset_prime_parameters()
        prime_params = self._get_prime_params()
        if len(prime_params) > 0:
            self._inner_optimizer = torch.optim.AdamW(
                prime_params,
                lr=self.ttt_inner_lr,
                betas=(self.ttt_inner_b1, self.ttt_inner_b2),
                weight_decay=0.0,
            )
        else:
            self._inner_optimizer = None

    @torch.enable_grad()
    def _ttt_step(
        self,
        denoised_pred: torch.Tensor,
        conditional_dict: dict,
        current_start: int,
    ):
        """One TTT inner-loop step: diffusion loss on *denoised_pred*, AdamW on prime params."""
        if self._inner_optimizer is None:
            return

        batch_size, num_frames = denoised_pred.shape[:2]
        device = denoised_pred.device

        pseudo_gt = denoised_pred.detach()
        ttt_timestep = torch.randint(
            self.ttt_min_timestep, self.ttt_max_timestep,
            [batch_size, num_frames], device=device, dtype=torch.long)
        ttt_noise = torch.randn_like(pseudo_gt)
        noisy_input = self.scheduler.add_noise(
            pseudo_gt.flatten(0, 1),
            ttt_noise.flatten(0, 1),
            ttt_timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frames))

        flow_pred, _ = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=ttt_timestep,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=current_start * self.frame_seq_length,
            update_cache=False,
        )

        target = ttt_noise - pseudo_gt
        loss = torch.mean((flow_pred.float() - target.float()) ** 2)

        self._inner_optimizer.zero_grad()
        loss.backward()
        if self.ttt_inner_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self._get_prime_params(), self.ttt_inner_clip)
        self._inner_optimizer.step()

    # ------------------------------------------------------------------
    # Override inference to inject TTT steps
    # ------------------------------------------------------------------
    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        # Each video starts from fresh zero-initialised prime adapters + optimizer
        self._reset_prime_and_optimizer()

        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        if low_memory:
            from demo_utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype)

        # Step 1: Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length)
                current_start_frame += 1
            else:
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref = initial_latent[
                    :, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref
                self.generator(
                    noisy_image_or_video=current_ref,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length)
                current_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, current_start_frame - num_input_frames:
                current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device, dtype=torch.int64) * current_timestep
                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length)
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames],
                            device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length)

            # Step 3.2: Record output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.2b: TTT inner-loop update on generated block
            self._ttt_step(denoised_pred, conditional_dict, current_start_frame)

            # Step 3.3: Rerun with context noise to update KV cache
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length)

            # Step 3.4: Advance
            current_start_frame += current_num_frames

        # Step 4: Decode
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video
