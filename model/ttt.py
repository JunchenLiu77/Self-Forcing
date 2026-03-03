"""TTT-E2E model for Self-Forcing video generation.

Extends DMD with meta-learned test-time training: the outer loop
(AdamW) optimises the model so that inner-loop AdamW steps on the
PrimeFFN adapters at test time yield good predictions.
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import torch.distributed as dist

from model.dmd import DMD
from pipeline.ttt_training import TTTTrainingPipeline


class TTT_E2E(DMD):
    """DMD + end-to-end meta-learned TTT inner loop."""

    def __init__(self, args, device):
        super().__init__(args, device)
        self.ttt_inner_lr = getattr(args, "ttt_inner_lr", 1e-3)
        self.ttt_inner_clip = getattr(args, "ttt_inner_clip", 1.0)
        self.ttt_min_timestep = getattr(args, "ttt_min_timestep", 20)
        self.ttt_max_timestep = getattr(args, "ttt_max_timestep", 980)
        self.ttt_inner_b1 = getattr(args, "ttt_inner_b1", 0.9)
        self.ttt_inner_b2 = getattr(args, "ttt_inner_b2", 0.999)

    def _initialize_inference_pipeline(self):
        self.inference_pipeline = TTTTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            ttt_inner_lr=self.ttt_inner_lr,
            ttt_inner_clip=self.ttt_inner_clip,
            ttt_min_timestep=self.ttt_min_timestep,
            ttt_max_timestep=self.ttt_max_timestep,
            ttt_inner_b1=self.ttt_inner_b1,
            ttt_inner_b2=self.ttt_inner_b2,
        )

    def _get_prime_params(self):
        """Resolve prime params from possibly FSDP-wrapped generator."""
        model = self.generator.model
        if hasattr(model, "_fsdp_wrapped_module"):
            model = model._fsdp_wrapped_module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model.get_prime_parameter_list()

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor = None,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Generate video with TTT inner loop, then compute DMD loss.

        The inner loop SGD steps use ``create_graph=True`` so the outer
        optimiser can back-propagate through them (meta-learning).
        """
        # Step 1: Unroll generator with TTT inner-loop updates
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = \
            self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
            )

        # Step 2: Compute DMD loss as usual
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )

        # Step 3: Also collect the accumulated TTT inner loss for logging
        ttt_loss = getattr(self.inference_pipeline, '_last_ttt_loss', torch.tensor(0.0))
        dmd_log_dict["ttt_inner_loss"] = ttt_loss.detach()

        return dmd_loss, dmd_log_dict
