# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script demonstrates how to train Diffusion Policy on the PushT environment."""

import argparse
import math
import time
from itertools import cycle
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
import wandb
from diffusers.optimization import get_scheduler

def apply_equivariant_aug(batch, max_translate: float = 150.0, arena_size: float = 512.0):
    """Equivariant rigid-body augmentation for PushT.

    Applies the same random rotation + translation to all modalities:
    - observation.image  (B, T_obs, C, H, W)  float32 [0, 1]
    - observation.state  (B, T_obs, 2)         agent_pos in [0, 512]
    - action             (B, T_act, 2)         target pos in [0, 512]

    Because every modality sees the same transform, the learning signal stays
    consistent: the policy learns to push the T-block to wherever the green
    silhouette appears in the image, rather than to a fixed pixel location.

    Must be called BEFORE the preprocessor (which normalises values).
    """
    imgs = batch["observation.image"]          # (B, T_obs, C, H, W)
    B, T_obs, C, H, W = imgs.shape

    # Sample one rigid transform per batch item
    angles_deg = torch.FloatTensor(B).uniform_(-180.0, 180.0)
    tx_coord    = torch.FloatTensor(B).uniform_(-max_translate, max_translate)
    ty_coord    = torch.FloatTensor(B).uniform_(-max_translate, max_translate)

    # Pixel-space translation: scale from [0,512] coords to image pixels
    tx_px = (tx_coord * H / arena_size).tolist()
    ty_px = (ty_coord * H / arena_size).tolist()

    # Augment images sample-by-sample (TF.affine works on individual CHW tensors)
    imgs_aug = torch.zeros_like(imgs)
    for b in range(B):
        for t in range(T_obs):
            imgs_aug[b, t] = TF.affine(
                imgs[b, t],
                angle=angles_deg[b].item(),
                translate=[tx_px[b], ty_px[b]],
                scale=1.0,
                shear=0,
                fill=[1.0, 1.0, 1.0],   # white = PushT background colour
            )
    batch["observation.image"] = imgs_aug

    # Rotate + translate 2-D coordinates.
    # PushT uses image coords (y-axis points DOWN), same as TF.affine convention:
    #   x' = (x - 256)*cos + (y - 256)*sin + 256 + tx
    #   y' = -(x - 256)*sin + (y - 256)*cos + 256 + ty
    dev = batch["observation.state"].device
    theta  = (angles_deg * (math.pi / 180.0))
    cos_t  = theta.cos().to(dev)[:, None]   # (B, 1)
    sin_t  = theta.sin().to(dev)[:, None]
    tx_c   = tx_coord.to(dev)[:, None]
    ty_c   = ty_coord.to(dev)[:, None]

    def _transform(coords):
        # coords: (B, T, 2)
        cx = coords[..., 0] - 256.0
        cy = coords[..., 1] - 256.0
        x_new = cx * cos_t - cy * sin_t + 256.0 + tx_c
        y_new = cx * sin_t + cy * cos_t + 256.0 + ty_c
        return torch.stack([x_new, y_new], dim=-1)

    batch["observation.state"] = _transform(batch["observation.state"])
    batch["action"]            = _transform(batch["action"])
    return batch


from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def run_eval(
    policy,
    preprocessor,
    postprocessor,
    device,
    n_episodes=10,
    observation_height=96,
    observation_width=96,
    seed_offset=0,
):
    """Run evaluation episodes and return metrics."""
    import gymnasium as gym
    import gym_pusht  # noqa: F401

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=observation_width,
        observation_height=observation_height,
    )

    policy.eval()
    successes, sum_rewards, max_rewards = [], [], []
    t_start = time.perf_counter()

    for ep in range(n_episodes):
        policy.reset()
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        ep_reward, ep_max_reward, success = 0.0, 0.0, False

        while not done:
            obs_dict = preprocess_observation(obs)
            obs_dict = preprocessor(obs_dict)
            with torch.inference_mode():
                action = policy.select_action(obs_dict)
            action = postprocessor(action)
            action_np = action.to("cpu").numpy()
            if action_np.ndim == 2:
                action_np = action_np[0]
            obs, reward, terminated, truncated, info = env.step(action_np)
            ep_reward += reward
            ep_max_reward = max(ep_max_reward, reward)
            success = info.get("is_success", False)
            done = terminated or truncated

        successes.append(success)
        sum_rewards.append(ep_reward)
        max_rewards.append(ep_max_reward)

    env.close()
    eval_s = time.perf_counter() - t_start
    policy.train()

    return {
        "eval/pc_success": sum(successes) / n_episodes * 100,
        "eval/avg_sum_reward": sum(sum_rewards) / n_episodes,
        "eval/avg_max_reward": sum(max_rewards) / n_episodes,
        "eval/eval_s": eval_s,
        "eval/n_episodes": n_episodes,
    }


def save_checkpoint(
    ckpt_dir, policy, preprocessor, postprocessor, optimizer, lr_scheduler, step
):
    """Save policy weights, preprocessors, and training state."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    preprocessor.save_pretrained(ckpt_dir)
    postprocessor.save_pretrained(ckpt_dir)
    # Save optimizer, scheduler state and current step for resume.
    torch.save(
        {
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        },
        ckpt_dir / "training_state.pt",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from a checkpoint."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default="outputs/train/pusht_diffusion_100000_steps",
        help="Path to checkpoint directory to resume from. Required when --resume is set.",
    )
    args = parser.parse_args()

    if args.resume and args.checkpoint_path is None:
        parser.error("--checkpoint_path is required when --resume is set.")

    output_directory = Path("outputs/train/pusht_diffusion_200000_steps")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")

    training_steps = 200000
    log_freq = 200
    eval_freq = 25000
    save_freq = 25000
    eval_episodes = 10

    # ── Dataset & policy config ──────────────────────────────────────────────
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }
    cfg = DiffusionConfig(
        input_features=input_features, output_features=output_features
    )

    # ── Policy ───────────────────────────────────────────────────────────────
    if args.resume:
        print(f"Resuming from checkpoint: {args.checkpoint_path}")
        policy = DiffusionPolicy.from_pretrained(args.checkpoint_path)
    else:
        policy = DiffusionPolicy(cfg)

    policy.train()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=dataset_metadata.stats
    )

    # ── Dataset & dataloader ─────────────────────────────────────────────────
    delta_timestamps = {
        "observation.image": [
            i / dataset_metadata.fps for i in cfg.observation_delta_indices
        ],
        "observation.state": [
            i / dataset_metadata.fps for i in cfg.observation_delta_indices
        ],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }
    dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=cfg.drop_n_last_frames,
        shuffle=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=1e-4,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
    )
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=training_steps,
    )

    # ── Resume: restore optimizer, scheduler, and step ───────────────────────
    start_step = 0
    if args.resume:
        state_path = args.checkpoint_path / "training_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(
                f"training_state.pt not found in {args.checkpoint_path}. "
                "Make sure the checkpoint was saved by this script."
            )
        state = torch.load(state_path, map_location=device)
        start_step = state["step"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])
        print(f"Resumed from step {state['step']} → continuing from step {start_step}")

    # ── WandB ────────────────────────────────────────────────────────────────
    wandb.init(
        project="pusht_diffusion",
        name=f"pusht_diffusion_200k{'_resume' if args.resume else ''}",
        resume="allow" if args.resume else None,
        config={
            "training_steps": training_steps,
            "batch_size": 64,
            "lr": 1e-4,
            "start_step": start_step,
            "eval_freq": eval_freq,
            "eval_episodes": eval_episodes,
        },
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    dl_iter = cycle(dataloader)

    for step in range(start_step, training_steps):
        t_update_start = time.perf_counter()

        batch = next(dl_iter)
        batch = preprocessor(batch)

        loss, _ = policy.forward(batch)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        update_s = time.perf_counter() - t_update_start

        if step % log_freq == 0:
            lr = lr_scheduler.get_last_lr()[0]
            print(
                f"step: {step:6d}  loss: {loss.item():.4f}  grad_norm: {grad_norm:.3f}  lr: {lr:.2e}"
            )
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/grad_norm": float(grad_norm),
                    "train/lr": lr,
                    "train/update_s": update_s,
                },
                step=step,
            )

        if eval_freq > 0 and (step + 1) % eval_freq == 0:
            print(f"\n--- Eval at step {step + 1} ---")
            eval_metrics = run_eval(
                policy,
                preprocessor,
                postprocessor,
                device,
                n_episodes=eval_episodes,
                seed_offset=0,
            )
            print(
                f"  pc_success={eval_metrics['eval/pc_success']:.1f}%"
                f"  avg_sum_reward={eval_metrics['eval/avg_sum_reward']:.2f}"
                f"  avg_max_reward={eval_metrics['eval/avg_max_reward']:.4f}"
                f"  eval_s={eval_metrics['eval/eval_s']:.1f}s"
            )
            wandb.log(eval_metrics, step=step + 1)

        if save_freq > 0 and (step + 1) % save_freq == 0:
            ckpt_dir = output_directory / f"checkpoint_{step + 1:06d}"
            save_checkpoint(
                ckpt_dir,
                policy,
                preprocessor,
                postprocessor,
                optimizer,
                lr_scheduler,
                step,
            )
            print(f"  → checkpoint saved: {ckpt_dir}\n")

    # ── Final save ────────────────────────────────────────────────────────────
    save_checkpoint(
        output_directory,
        policy,
        preprocessor,
        postprocessor,
        optimizer,
        lr_scheduler,
        training_steps - 1,
    )
    print(f"\nFinal checkpoint saved to {output_directory}")
    wandb.finish()


if __name__ == "__main__":
    main()
