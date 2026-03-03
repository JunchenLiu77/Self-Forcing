"""TTT-E2E training pipeline.

Extends SelfForcingTrainingPipeline with per-block inner-loop AdamW
updates on the PrimeFFN adapters.

Uses ``torch.func.functional_call`` for inner-loop weight updates so
the computation graph stays intact and the outer loop can
back-propagate through the inner steps (meta-learning).
"""

from typing import Dict, List, Optional
import torch
import torch.distributed as dist
from torch.func import functional_call

from pipeline.self_forcing_training import SelfForcingTrainingPipeline
from utils.wan_wrapper import WanDiffusionWrapper


class TTTTrainingPipeline(SelfForcingTrainingPipeline):

    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler,
        generator: WanDiffusionWrapper,
        num_frame_per_block=3,
        independent_first_frame: bool = False,
        same_step_across_blocks: bool = False,
        last_step_only: bool = False,
        num_max_frames: int = 21,
        context_noise: int = 0,
        ttt_inner_lr: float = 1e-3,
        ttt_inner_clip: float = 1.0,
        ttt_min_timestep: int = 20,
        ttt_max_timestep: int = 980,
        ttt_inner_b1: float = 0.9,
        ttt_inner_b2: float = 0.999,
        **kwargs,
    ):
        super().__init__(
            denoising_step_list=denoising_step_list,
            scheduler=scheduler,
            generator=generator,
            num_frame_per_block=num_frame_per_block,
            independent_first_frame=independent_first_frame,
            same_step_across_blocks=same_step_across_blocks,
            last_step_only=last_step_only,
            num_max_frames=num_max_frames,
            context_noise=context_noise,
            **kwargs,
        )
        self.ttt_inner_lr = ttt_inner_lr
        self.ttt_inner_clip = ttt_inner_clip
        self.ttt_min_timestep = ttt_min_timestep
        self.ttt_max_timestep = ttt_max_timestep
        self.ttt_inner_b1 = ttt_inner_b1
        self.ttt_inner_b2 = ttt_inner_b2
        self._last_ttt_loss = torch.tensor(0.0)

        # Functional inner-loop state (reset per trajectory)
        self._prime_overrides: Dict[str, torch.Tensor] = {}
        self._adam_m: Dict[str, torch.Tensor] = {}
        self._adam_v: Dict[str, torch.Tensor] = {}
        self._adam_t: int = 0

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
        """Reset the output layer of every PrimeFFN to zero and clear
        the functional override dict + Adam moment buffers."""
        self._unwrap_model().reset_prime_parameters()
        self._prime_overrides = {}
        self._adam_m = {}
        self._adam_v = {}
        self._adam_t = 0

    def _get_prime_named(self) -> Dict[str, torch.Tensor]:
        """Return ``{name: param}`` for all PrimeFFN params.

        If the functional override dict already contains updated tensors
        for those names, return those instead (they are the "current"
        inner-loop values and sit in the computation graph).
        """
        base = self._unwrap_model().get_prime_named_parameters()
        merged = {}
        for k, v in base.items():
            merged[k] = self._prime_overrides.get(k, v)
        return merged

    # ------------------------------------------------------------------
    def _generator_forward_with_overrides(self, **fwd_kwargs):
        """Run ``self.generator`` but with functional prime-param overrides.

        If ``_prime_overrides`` is non-empty, uses
        ``functional_call`` on the inner CausalWanModel so the
        forward pass uses the (graph-connected) override tensors
        instead of the actual ``nn.Parameter`` buffers.
        """
        if not self._prime_overrides:
            return self.generator(**fwd_kwargs)

        # Prefix override keys with "model." to match the WanDiffusionWrapper
        # namespace (generator.model = CausalWanModel).
        wrapper_overrides = {
            "model." + k: v for k, v in self._prime_overrides.items()
        }
        return functional_call(
            self.generator, wrapper_overrides, kwargs=fwd_kwargs)

    # ------------------------------------------------------------------
    @torch.enable_grad()
    def _ttt_inner_step(
        self,
        denoised_pred: torch.Tensor,
        conditional_dict: dict,
        current_start_frame: int,
    ) -> torch.Tensor:
        """Compute diffusion loss and functionally update prime params.

        Creates **new tensors** for the updated prime values (no
        in-place mutation) so the autograd graph stays valid for the
        outer loop.
        """
        prime_named = self._get_prime_named()
        if len(prime_named) == 0:
            return torch.tensor(0.0, device=denoised_pred.device)

        prime_names = list(prime_named.keys())
        prime_values = list(prime_named.values())

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

        # Forward with current prime overrides
        flow_pred, _ = self._generator_forward_with_overrides(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=ttt_timestep,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=current_start_frame * self.frame_seq_length,
            update_cache=False,
        )

        target = ttt_noise - pseudo_gt
        loss = torch.mean((flow_pred.float() - target.float()) ** 2)

        grads = torch.autograd.grad(
            loss, prime_values, create_graph=True, allow_unused=True)

        # Global norm clipping
        clip_coef = torch.tensor(1.0, device=device)
        if self.ttt_inner_clip > 0:
            grad_tensors = [g for g in grads if g is not None]
            if grad_tensors:
                total_norm = torch.sqrt(
                    sum(g.float().pow(2).sum() for g in grad_tensors))
                clip_coef = torch.clamp(
                    self.ttt_inner_clip / (total_norm + 1e-6), max=1.0)

        # AdamW update — functional (creates new tensors, no in-place)
        self._adam_t += 1
        b1, b2, eps = self.ttt_inner_b1, self.ttt_inner_b2, 1e-8
        bc1 = 1 - b1 ** self._adam_t
        bc2 = 1 - b2 ** self._adam_t

        new_overrides = {}
        for name, val, g in zip(prime_names, prime_values, grads):
            if g is None:
                new_overrides[name] = val
                continue
            g_clipped = g.float() * clip_coef

            m_prev = self._adam_m.get(name, torch.zeros_like(val))
            v_prev = self._adam_v.get(name, torch.zeros_like(val))
            m_new = b1 * m_prev + (1 - b1) * g_clipped
            v_new = b2 * v_prev + (1 - b2) * g_clipped * g_clipped
            self._adam_m[name] = m_new.detach()
            self._adam_v[name] = v_new.detach()

            m_hat = m_new / bc1
            v_hat = v_new / bc2
            # Functional update: new tensor, no mutation
            new_overrides[name] = val - self.ttt_inner_lr * m_hat / (v_hat.sqrt() + eps)

        self._prime_overrides = new_overrides
        return loss

    # ------------------------------------------------------------------
    def _apply_overrides_to_params(self):
        """Copy current override values back into the actual nn.Parameters.

        Called before the context-caching forward pass (which runs under
        torch.no_grad and needs the real params to reflect the inner-loop
        state).  This is safe because context caching is always no_grad.
        """
        if not self._prime_overrides:
            return
        model = self._unwrap_model()
        named = model.get_prime_named_parameters()
        with torch.no_grad():
            for k, p in named.items():
                if k in self._prime_overrides:
                    p.copy_(self._prime_overrides[k].detach())

    # ------------------------------------------------------------------
    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        initial_latent: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
        **conditional_dict,
    ) -> torch.Tensor:
        """Autoregressive generation with TTT inner-loop updates per block."""
        self._reset_prime_and_optimizer()

        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (
            self.independent_first_frame and initial_latent is not None
        ):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype)

        # Step 1: Init caches
        self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)

        # Step 2: Cache context
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length)
            current_start_frame += 1

        # Step 3: Denoising loop with TTT
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(
            len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21

        accumulated_ttt_loss = torch.tensor(0.0, device=noise.device)
        num_ttt_steps = 0

        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame - num_input_frames:
                current_start_frame + current_num_frames - num_input_frames]

            # Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device, dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self._generator_forward_with_overrides(
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
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            _, denoised_pred = self._generator_forward_with_overrides(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length)
                    else:
                        _, denoised_pred = self._generator_forward_with_overrides(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length)
                    break

            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # TTT inner-loop step (functional: no in-place mutation)
            ttt_loss = self._ttt_inner_step(
                denoised_pred, conditional_dict, current_start_frame)
            accumulated_ttt_loss = accumulated_ttt_loss + ttt_loss
            num_ttt_steps += 1

            # Write overrides into real params for context caching
            self._apply_overrides_to_params()

            # Re-run with context noise to update cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            denoised_pred_ctx = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames],
                    device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred_ctx,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length)

            current_start_frame += current_num_frames

        if num_ttt_steps > 0:
            self._last_ttt_loss = accumulated_ttt_loss / num_ttt_steps
        else:
            self._last_ttt_loss = torch.tensor(0.0, device=noise.device)

        # Return timestep info
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
