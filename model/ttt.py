"""TTT-E2E model for Self-Forcing video generation.

Meta-learning variant: PrimeFFN "fast weights" are cloned per video,
updated functionally per block (inner loop with ``create_graph=True``),
and the outer DMD loss back-props through the entire inner-loop chain
to train both slow weights and the initial fast weights end-to-end.
"""

from typing import Tuple
import torch

from model.dmd import DMD
from pipeline import SelfForcingTrainingPipeline


class TTT_E2E(DMD):

    def __init__(self, args, device):
        super().__init__(args, device)
        self.ttt_inner_lr = getattr(args, "ttt_inner_lr", 1e-3)
        self.ttt_inner_clip = getattr(args, "ttt_inner_clip", 1.0)
        self.ttt_first_order = getattr(args, "ttt_first_order", False)

    def _initialize_inference_pipeline(self):
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            ttt_enabled=True,
            ttt_inner_lr=self.ttt_inner_lr,
            ttt_inner_clip=self.ttt_inner_clip,
            ttt_first_order=self.ttt_first_order,
        )

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor = None,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Generate video with meta-learning inner loop, then compute DMD loss.

        The pipeline's ``inference_with_trajectory`` now produces a
        trajectory whose computational graph threads through the
        functional fast-weight updates.  ``super().generator_loss``
        computes the DMD loss on this trajectory; gradients flow back
        through the inner loop to both slow and initial-fast parameters.
        """
        dmd_loss, dmd_log_dict = super().generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=initial_latent,
        )

        ttt_loss = getattr(self.inference_pipeline, '_last_ttt_loss', torch.tensor(0.0))
        dmd_log_dict["ttt_inner_loss"] = ttt_loss.detach()

        return dmd_loss, dmd_log_dict
