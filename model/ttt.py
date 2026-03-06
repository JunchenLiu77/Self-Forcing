"""TTT-E2E model for Self-Forcing video generation.

Extends DMD with test-time training: the generator adapts its PrimeFFN
adapters per-block during autoregressive rollout using a diffusion
(flow-prediction) self-supervised loss.  The outer DMD loss trains all
parameters end-to-end.

Use ``min_ttt_blocks`` / ``max_ttt_blocks`` to control how many warmup
blocks (with TTT inner-loop updates) are prepended before the 21-frame
gradient window each iteration.
"""

from typing import Tuple
import torch
import torch.distributed as dist

from model.dmd import DMD
from pipeline import SelfForcingTrainingPipeline


class TTT_E2E(DMD):

    def __init__(self, args, device):
        super().__init__(args, device)
        self.ttt_inner_lr = getattr(args, "ttt_inner_lr", 1e-3)
        self.ttt_inner_clip = getattr(args, "ttt_inner_clip", 1.0)
        self.ttt_inner_b1 = getattr(args, "ttt_inner_b1", 0.9)
        self.ttt_inner_b2 = getattr(args, "ttt_inner_b2", 0.999)

        min_ttt_blocks = getattr(args, "min_ttt_blocks", 0)
        max_ttt_blocks = getattr(args, "max_ttt_blocks", 0)
        if max_ttt_blocks > 0:
            base_frames = 21
            self.min_training_frames = base_frames + min_ttt_blocks * self.num_frame_per_block
            self.num_training_frames = base_frames + max_ttt_blocks * self.num_frame_per_block

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
            ttt_inner_b1=self.ttt_inner_b1,
            ttt_inner_b2=self.ttt_inner_b2,
        )

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor = None,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Generate video with TTT inner loop, then compute DMD loss."""
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
