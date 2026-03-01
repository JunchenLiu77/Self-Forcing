import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, default=None,
                    help="Output folder. If not set, auto-derived from checkpoint_path as videos/{exp_name}/{iter}/")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--first_n", type=int, default=0,
                    help="Only process the first N prompts (0 = process all)")
parser.add_argument("--num_windows", type=int, default=1,
                    help="Number of sliding windows for longer video generation (each adds ~72 pixel frames)")
args = parser.parse_args()

if args.output_folder is None and args.checkpoint_path:
    # Derive output folder from checkpoint path:
    # e.g. .../outputs/self_forcing_dmd/checkpoint_model_000350/model.pt
    #   -> exp_name = "self_forcing_dmd", iter = "000350"
    #   -> output_folder = "videos/self_forcing_dmd/000350"
    ckpt_dir = os.path.dirname(args.checkpoint_path)          # .../checkpoint_model_000350
    ckpt_dir_name = os.path.basename(ckpt_dir)                # checkpoint_model_000350
    exp_name = os.path.basename(os.path.dirname(ckpt_dir))    # self_forcing_dmd
    iter_str = ckpt_dir_name.rsplit("_", 1)[-1]               # 000350
    args.output_folder = os.path.join("videos", exp_name, iter_str)
    print(f"Auto-derived output folder: {args.output_folder}")

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    import re
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    key = 'generator' if not args.use_ema else 'generator_ema'
    ckpt = state_dict[key]
    pattern = re.compile(r'_fsdp_wrapped_module\.|_checkpoint_wrapped_module\.|_orig_mod\.')
    ckpt = {pattern.sub('', k): v for k, v in ckpt.items()}
    pipeline.generator.load_state_dict(ckpt)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


num_frame_per_block = getattr(config, 'num_frame_per_block', 1)

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    if args.first_n > 0 and i >= args.first_n:
        break

    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]

    if args.i2v:
        prompt = batch['prompts'][0]
    else:
        prompt = batch['prompts'][0]

    # Build output paths and check if all already exist
    win_tag = f"w{args.num_windows}"
    output_paths = {}
    for seed_idx in range(args.num_samples):
        if args.save_with_index:
            fname = f'{idx}_{seed_idx}_{win_tag}.mp4'
        else:
            fname = f'{prompt[:100]}_{seed_idx}_{win_tag}.mp4'
        output_paths[seed_idx] = os.path.join(args.output_folder, fname)

    if idx < num_prompts and all(os.path.exists(p) for p in output_paths.values()):
        print(f"Skipping idx={idx}, all outputs exist")
        continue

    # Prepare prompts and initial_latent
    if args.i2v:
        prompts = [prompt] * args.num_samples
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)
    else:
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        initial_latent = None

    all_video = []
    for window_idx in range(args.num_windows):
        if initial_latent is not None:
            num_initial = initial_latent.shape[1]
            noise_frames = ((args.num_output_frames - num_initial) // num_frame_per_block) * num_frame_per_block
        else:
            noise_frames = args.num_output_frames

        sampled_noise = torch.randn(
            [args.num_samples, noise_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent,
            low_memory=low_memory,
        )

        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        if window_idx > 0:
            overlap_pixels = (num_frame_per_block - 1) * 4 + 1
            current_video = current_video[:, overlap_pixels:]
        all_video.append(current_video)

        if window_idx < args.num_windows - 1:
            initial_latent = latents[:, -num_frame_per_block:].to(dtype=torch.bfloat16)

        pipeline.vae.model.clear_cache()

    video = 255.0 * torch.cat(all_video, dim=1)

    if idx < num_prompts:
        for seed_idx in range(args.num_samples):
            output_path = output_paths[seed_idx]
            if os.path.exists(output_path):
                continue
            write_video(output_path, video[seed_idx], fps=16)
