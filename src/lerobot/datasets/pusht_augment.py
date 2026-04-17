"""
PushT Dataset Geometric Augmentation
=====================================
对 PushT 数据集进行旋转、平移等几何增强，保证图像与 state/action 坐标的一致性。

坐标系说明：
  - 图像空间  : 96×96 像素，原点左上角，y 轴向下
  - 坐标空间  : state/action 约在 [0, 512]×[0, 512]
  - 缩放比    : SCALE = 96/512 ≈ 0.1875
  - 图像中心  : (48, 48)  ↔  坐标中心 (256, 256)

旋转方向约定（TF.affine 正角 = CW 顺时针）：
  由于图像 y 轴向下，与 TF.affine CW 旋转一致的坐标变换为：
    x' = cx + (x-cx)*cos θ - (y-cy)*sin θ
    y' = cy + (x-cx)*sin θ + (y-cy)*cos θ
  验证：(dx=1,dy=0), θ=90° → x'=0, y'=1（右 → 下，视觉 CW）✓
"""

import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import default_collate
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import DEFAULT_FEATURES

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 96
COORD_SIZE = 512
SCALE = IMG_SIZE / COORD_SIZE  # 0.1875

CX_COORD = COORD_SIZE / 2  # 256.0
CY_COORD = COORD_SIZE / 2  # 256.0


# ─── Core: RigidTransform ─────────────────────────────────────────────────────


class RigidTransform:
    """2D rigid-body transform: rotation around the scene center followed by translation.

    All image and coordinate operations are fused into a single step so that
    the floating-point representation is identical for both modalities.

    Convention (matches TF.affine with positive angle = CW on screen):
        x' = (x - cx)*cos - (y - cy)*sin + cx + tx_coord
        y' = (x - cx)*sin + (y - cy)*cos + cy + ty_coord

    Args:
        angle_deg : rotation in degrees (positive = CW on screen)
        tx_coord  : x-translation in coordinate space [0, 512]
        ty_coord  : y-translation in coordinate space [0, 512]
    """

    def __init__(self, angle_deg: float, tx_coord: float, ty_coord: float):
        self.angle_deg = angle_deg
        self.tx_coord = tx_coord
        self.ty_coord = ty_coord
        theta = math.radians(angle_deg)
        self.cos_t = math.cos(theta)
        self.sin_t = math.sin(theta)

    # ── Coordinate transform ──────────────────────────────────────────────────

    def apply_coords(self, xy):
        """Apply to (..., 2) torch.Tensor or np.ndarray in coordinate space."""
        if isinstance(xy, torch.Tensor):
            dx = xy[..., 0] - CX_COORD
            dy = xy[..., 1] - CY_COORD
            x_new = dx * self.cos_t - dy * self.sin_t + CX_COORD + self.tx_coord
            y_new = dx * self.sin_t + dy * self.cos_t + CY_COORD + self.ty_coord
            return torch.stack([x_new, y_new], dim=-1)
        else:
            xy = np.asarray(xy, dtype=float)
            dx = xy[..., 0] - CX_COORD
            dy = xy[..., 1] - CY_COORD
            x_new = dx * self.cos_t - dy * self.sin_t + CX_COORD + self.tx_coord
            y_new = dx * self.sin_t + dy * self.cos_t + CY_COORD + self.ty_coord
            return np.stack([x_new, y_new], axis=-1)

    def is_valid(self, xy) -> bool:
        """Return True if every coord in (..., 2) lies within [0, COORD_SIZE]."""
        if isinstance(xy, torch.Tensor):
            return bool(
                (xy[..., 0] >= 0).all()
                and (xy[..., 0] <= COORD_SIZE).all()
                and (xy[..., 1] >= 0).all()
                and (xy[..., 1] <= COORD_SIZE).all()
            )
        else:
            xy = np.asarray(xy)
            return bool(
                (xy[..., 0] >= 0).all()
                and (xy[..., 0] <= COORD_SIZE).all()
                and (xy[..., 1] >= 0).all()
                and (xy[..., 1] <= COORD_SIZE).all()
            )

    # ── Image transform ───────────────────────────────────────────────────────

    def apply_image(self, img: torch.Tensor, border: int = 3) -> torch.Tensor:
        """Apply to (C, H, W) float32 [0, 1] image.

        Before transforming, the outer ``border`` pixels of the image are set
        to pure white. This ensures that bilinear interpolation at the content
        boundary always blends white with white, producing a clean white edge
        rather than a gray fringe.
        """
        H, W = img.shape[-2], img.shape[-1]

        # ── Step 1: whiten the outer border pixels ────────────────────────────
        img = img.clone()
        img[:, :border, :] = 1.0   # top
        img[:, -border:, :] = 1.0  # bottom
        img[:, :, :border] = 1.0   # left
        img[:, :, -border:] = 1.0  # right

        # ── Step 2: pad + affine + crop ───────────────────────────────────────
        pad = int(math.ceil(max(H, W) * 0.3))
        tx_px = self.tx_coord * (H / COORD_SIZE)
        ty_px = self.ty_coord * (H / COORD_SIZE)
        padded = TF.pad(img, padding=pad, fill=1.0)
        out = TF.affine(
            padded,
            angle=self.angle_deg,
            translate=[tx_px, ty_px],
            scale=1.0,
            shear=0,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=1.0,
        )
        return TF.center_crop(out, [H, W])

    # ── Factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def identity() -> "RigidTransform":
        return RigidTransform(0.0, 0.0, 0.0)

    @staticmethod
    def sample_valid(
        state,
        action,
        max_angle: float = 180.0,
        max_trans: float = 100.0,
        max_try: int = 20,
        rng=None,
    ) -> "RigidTransform":
        """Sample a random rigid transform that keeps all state/action coords in bounds.

        Falls back to the identity transform if no valid sample is found within
        ``max_try`` attempts.

        Args:
            state    : (..., 2) coords for the agent position
            action   : (..., 2) coords for the target position
            max_angle: uniform range [-max_angle, +max_angle] in degrees
            max_trans: uniform range [-max_trans, +max_trans] in coord units
            max_try  : number of rejection-sampling attempts before fallback
            rng      : numpy Generator (created if None)
        """
        _rng = rng if rng is not None else np.random.default_rng()
        for _ in range(max_try):
            angle = float(_rng.uniform(-max_angle, max_angle))
            tx = float(_rng.uniform(-max_trans, max_trans))
            ty = float(_rng.uniform(-max_trans, max_trans))
            tfm = RigidTransform(angle, tx, ty)
            if tfm.is_valid(tfm.apply_coords(state)) and tfm.is_valid(
                tfm.apply_coords(action)
            ):
                return tfm
        return RigidTransform.identity()


# ─── Single-frame augmentation ────────────────────────────────────────────────


class PushTAugmentation:
    """Apply a RigidTransform consistently to a single LeRobot frame dict.

    Useful for on-the-fly augmentation, unit tests, and visualization.

    Usage:
        aug = PushTAugmentation(angle_deg=30, tx_coord=20, ty_coord=-15)
        aug_frame = aug(frame)

        # Or wrap a pre-built transform:
        aug = PushTAugmentation.from_transform(tfm)
        aug_frame = aug(frame)
    """

    def __init__(
        self,
        angle_deg: float = 0.0,
        tx_coord: float = 0.0,
        ty_coord: float = 0.0,
    ):
        self.tfm = RigidTransform(angle_deg, tx_coord, ty_coord)

    @classmethod
    def from_transform(cls, tfm: RigidTransform) -> "PushTAugmentation":
        obj = cls.__new__(cls)
        obj.tfm = tfm
        return obj

    def __call__(self, frame: dict) -> dict:
        aug = dict(frame)
        aug["observation.image"] = self.tfm.apply_image(frame["observation.image"])
        aug["observation.state"] = self.tfm.apply_coords(
            frame["observation.state"].clone()
        )
        aug["action"] = self.tfm.apply_coords(frame["action"].clone())
        return aug


# ─── Batch-level augmentation (for DataLoader collate_fn) ─────────────────────


def apply_equivariant_aug(
    batch: dict,
    max_angle_deg: float = 180.0,
    max_translate_coord: float = 150.0,
    arena_size: float = float(COORD_SIZE),
) -> dict:
    """Equivariant rigid-body augmentation applied to a collated batch.

    One independent RigidTransform is sampled per batch element via
    rejection sampling.  The same transform is applied to every modality:
      - observation.image  (B, T_obs, C, H, W)  float32 [0, 1]
      - observation.state  (B, T_obs, 2)         agent_pos in [0, 512]
      - action             (B, T_act, 2)         target_pos in [0, 512]

    Coordinate transforms are applied with vectorized tensor ops (no Python
    loop over B after sampling).  Image transforms still loop over B×T_obs
    because TF.affine does not support batched calls.

    Must be called BEFORE the preprocessor (which normalises values).
    """
    imgs = batch["observation.image"]  # (B, T_obs, C, H, W)
    B, T_obs, C, H, W = imgs.shape

    # ── Sample one RigidTransform per batch item ──────────────────────────────
    angles = torch.zeros(B)
    tx_c = torch.zeros(B)
    ty_c = torch.zeros(B)

    for b in range(B):
        tfm = RigidTransform.sample_valid(
            batch["observation.state"][b],
            batch["action"][b],
            max_angle=max_angle_deg,
            max_trans=max_translate_coord,
            max_try=10,
        )
        angles[b] = tfm.angle_deg
        tx_c[b] = tfm.tx_coord
        ty_c[b] = tfm.ty_coord

    # ── Augment images (loop; TF.affine is not batchable) ────────────────────
    pad = int(math.ceil(max(H, W) * 0.3))
    tx_px = (tx_c * H / arena_size).tolist()
    ty_px = (ty_c * H / arena_size).tolist()
    imgs_aug = torch.empty_like(imgs)
    for b in range(B):
        for t in range(T_obs):
            padded = TF.pad(imgs[b, t], padding=pad, fill=1.0)
            augmented = TF.affine(
                padded,
                angle=angles[b].item(),
                translate=[tx_px[b], ty_px[b]],
                scale=1.0,
                shear=0,
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=1.0,
            )
            imgs_aug[b, t] = TF.center_crop(augmented, [H, W])
    batch["observation.image"] = imgs_aug

    # ── Augment goal images (B, C, H, W) with the same per-sample transform ──
    for key in list(batch.keys()):
        if key.startswith("goal.") and isinstance(batch[key], torch.Tensor) and batch[key].ndim == 4:
            goal_imgs = batch[key]
            goal_imgs_aug = torch.empty_like(goal_imgs)
            for b in range(B):
                padded = TF.pad(goal_imgs[b], padding=pad, fill=1.0)
                augmented = TF.affine(
                    padded,
                    angle=angles[b].item(),
                    translate=[tx_px[b], ty_px[b]],
                    scale=1.0,
                    shear=0,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    fill=1.0,
                )
                goal_imgs_aug[b] = TF.center_crop(augmented, [H, W])
            batch[key] = goal_imgs_aug

    # ── Augment coordinates (vectorized) ─────────────────────────────────────
    dev = batch["observation.state"].device
    theta = angles * (math.pi / 180.0)
    cos_t = theta.cos().to(dev)[:, None]  # (B, 1) — broadcasts over T
    sin_t = theta.sin().to(dev)[:, None]
    tx = tx_c.to(dev)[:, None]
    ty = ty_c.to(dev)[:, None]

    def _transform_coords(coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, T, 2)
        dx = coords[..., 0] - CX_COORD
        dy = coords[..., 1] - CY_COORD
        x_new = dx * cos_t - dy * sin_t + CX_COORD + tx
        y_new = dx * sin_t + dy * cos_t + CY_COORD + ty
        return torch.stack([x_new, y_new], dim=-1)

    batch["observation.state"] = _transform_coords(batch["observation.state"])
    batch["action"] = _transform_coords(batch["action"])
    return batch


class AugmentCollate:
    """Picklable collate_fn that assembles samples and applies equivariant augmentation.

    Picklability is required when ``num_workers > 0`` so worker processes can
    receive this object via pickle.

    Usage:
        loader = DataLoader(
            dataset,
            batch_size=64,
            collate_fn=AugmentCollate(max_angle_deg=180, max_translate_coord=100),
            num_workers=4,
        )
    """

    def __init__(
        self,
        max_angle_deg: float = 180.0,
        max_translate_coord: float = 100.0,
    ):
        self.max_angle_deg = max_angle_deg
        self.max_translate_coord = max_translate_coord

    def __call__(self, samples: list) -> dict:
        batch = default_collate(samples)
        return apply_equivariant_aug(
            batch,
            max_angle_deg=self.max_angle_deg,
            max_translate_coord=self.max_translate_coord,
        )


# ─── Visualization ────────────────────────────────────────────────────────────


def _draw_frame(ax, frame: dict, title: str) -> None:
    """Draw image with overlaid state (green dot) and action (red arrow + triangle).

    Coordinate labels (in coord space [0, 512]) are printed next to each marker,
    offset slightly to avoid overlapping the marker itself.
    """
    img = frame["observation.image"].permute(1, 2, 0).numpy()
    ax.imshow(np.clip(img, 0.0, 1.0), origin="upper")

    state_coord = frame["observation.state"].numpy()  # in [0, 512]
    action_coord = frame["action"].numpy()  # in [0, 512]
    state_px = state_coord * SCALE  # in image pixels
    action_px = action_coord * SCALE

    # ── Markers + arrow ───────────────────────────────────────────────────────
    ax.plot(state_px[0], state_px[1], "go", markersize=7, zorder=5)
    ax.annotate(
        "",
        xy=action_px,
        xytext=state_px,
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        zorder=6,
    )
    ax.plot(action_px[0], action_px[1], "r^", markersize=7, zorder=5)

    # ── Coordinate labels ─────────────────────────────────────────────────────
    # Offset direction: push the label away from the image center so it doesn't
    # cover the marker.  A fixed 3-pixel nudge is enough at 96×96 resolution.
    label_kw = dict(
        fontsize=6,
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, ec="none"),
    )
    cx_px = IMG_SIZE / 2

    s_offset_x = 3 if state_px[0] >= cx_px else -3
    ax.text(
        state_px[0] + s_offset_x,
        state_px[1] - 3,
        f"s({state_coord[0]:.0f},{state_coord[1]:.0f})",
        color="green",
        ha="left" if s_offset_x > 0 else "right",
        va="bottom",
        **label_kw,
    )

    a_offset_x = 3 if action_px[0] >= cx_px else -3
    ax.text(
        action_px[0] + a_offset_x,
        action_px[1] + 4,
        f"a({action_coord[0]:.0f},{action_coord[1]:.0f})",
        color="red",
        ha="left" if a_offset_x > 0 else "right",
        va="top",
        **label_kw,
    )

    ax.set_title(title, fontsize=8, pad=3)
    ax.axis("off")


def visualize_augmentations(
    dataset,
    augmentations: list[tuple],
    n_frames: int = 4,
    episode_idx: int = 0,
    save_path: str = "pusht_augmentation.png",
) -> None:
    """Save a side-by-side comparison of original frames and augmented versions.

    Args:
        augmentations : list of (PushTAugmentation, label_str)
        n_frames      : number of frames sampled uniformly from the episode
        episode_idx   : which episode to visualise
        save_path     : output PNG path
    """
    ep = dataset.meta.episodes[episode_idx]
    from_idx = int(ep["dataset_from_index"])
    to_idx = int(ep["dataset_to_index"])
    frame_indices = (
        np.linspace(0, to_idx - from_idx - 1, n_frames, dtype=int) + from_idx
    )

    n_cols = 1 + len(augmentations)
    fig, axes = plt.subplots(
        n_frames, n_cols, figsize=(3.0 * n_cols, 3.2 * n_frames), squeeze=False
    )

    for row, fi in enumerate(frame_indices):
        frame = dataset[int(fi)]
        ts = float(frame["timestamp"])
        _draw_frame(axes[row, 0], frame, f"Original  t={ts:.2f}s")
        for col, (aug_fn, label) in enumerate(augmentations, start=1):
            _draw_frame(axes[row, col], aug_fn(frame), label)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="g",
            linestyle="None",
            markersize=7,
            label="state",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="r",
            linestyle="None",
            markersize=7,
            label="action",
        ),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=9)
    fig.suptitle(
        f"PushT Geometric Augmentation  (episode {episode_idx})", fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"[OK] 可视化已保存 → {save_path}")
    plt.close()


# ─── Dataset-level augmentation ───────────────────────────────────────────────


def create_augmented_dataset(
    source_repo_id: str = "lerobot/pusht",
    output_repo_id: str = "local/pusht_augmented",
    output_root: str | Path = "outputs/datasets/pusht_augmented",
    n_augments: int = 2,
    max_angle_deg: float = 180.0,
    max_translate_coord: float = 100.0,
    include_original: bool = True,
    seed: int = 42,
) -> LeRobotDataset:
    """Create a new LeRobotDataset by augmenting an existing one.

    For every source episode, one or more augmented copies are written with the
    SAME RigidTransform applied to every frame, keeping episodes spatially
    consistent.  The transform is chosen by rejection sampling (all state/action
    coords must stay within [0, 512] after the transform).

    Augmented modalities:
      - observation.image : pad → TF.affine(rotate+translate) → center_crop
      - observation.state : 2-D coord rotation + translation (coord space)
      - action            : same as state

    Scalar fields (next.reward, next.done, next.success) are copied unchanged
    because reward depends on relative positions, which are rigid-transform
    invariant.

    Args:
        source_repo_id      : HF or local dataset to augment
        output_repo_id      : repo_id label for the new dataset
        output_root         : directory where the new dataset is saved
        n_augments          : number of augmented copies per source episode
        max_angle_deg       : rotation range [-max, +max] degrees
        max_translate_coord : translation range [-max, +max] in coord units [0, 512]
        include_original    : also write the unaugmented episodes
        seed                : numpy random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    print(f"Loading source dataset: {source_repo_id}")
    src = LeRobotDataset(source_repo_id)
    user_features = {k: v for k, v in src.features.items() if k not in DEFAULT_FEATURES}

    output_root = Path(output_root)
    if output_root.exists():
        import shutil

        print(f"Removing existing directory: {output_root}")
        shutil.rmtree(output_root)

    print(f"Creating destination dataset at: {output_root}")
    dst = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=src.fps,
        features=user_features,
        root=output_root,
        use_videos=True,
        image_writer_threads=4,
        vcodec="libsvtav1",
    )

    passes = (["original"] if include_original else []) + [
        f"aug_{i}" for i in range(n_augments)
    ]
    total_episodes = src.num_episodes * len(passes)

    print(
        f"\nSource episodes : {src.num_episodes}\n"
        f"Passes per ep   : {len(passes)}  ({passes})\n"
        f"Total new eps   : {total_episodes}\n"
    )

    with tqdm(total=total_episodes, desc="Writing episodes", unit="ep") as pbar:
        for ep_idx in range(src.num_episodes):
            ep = src.meta.episodes[ep_idx]
            from_idx = int(ep["dataset_from_index"])
            to_idx = int(ep["dataset_to_index"])
            task = ep["tasks"][0]

            # Pre-load state/action for the whole episode (used for validity check)
            ep_states = torch.stack(
                [src[fi]["observation.state"] for fi in range(from_idx, to_idx)]
            )
            ep_actions = torch.stack(
                [src[fi]["action"] for fi in range(from_idx, to_idx)]
            )

            for pass_name in passes:
                # ── Sample transform for this episode ─────────────────────────
                if pass_name == "original":
                    tfm = RigidTransform.identity()
                else:
                    tfm = RigidTransform.sample_valid(
                        ep_states,
                        ep_actions,
                        max_angle=max_angle_deg,
                        max_trans=max_translate_coord,
                        max_try=20,
                        rng=rng,
                    )

                # ── Write every frame ─────────────────────────────────────────
                for fi in range(from_idx, to_idx):
                    frame = src[fi]

                    img = tfm.apply_image(frame["observation.image"])
                    img_np = (
                        (img.permute(1, 2, 0) * 255)
                        .clamp(0, 255)
                        .to(torch.uint8)
                        .numpy()
                    )
                    state = tfm.apply_coords(frame["observation.state"].numpy()).astype(
                        np.float32
                    )
                    action = tfm.apply_coords(frame["action"].numpy()).astype(
                        np.float32
                    )

                    dst.add_frame(
                        {
                            "task": task,
                            "observation.image": img_np,
                            "observation.state": state,
                            "action": action,
                            "next.reward": np.array(
                                [float(frame["next.reward"])], dtype=np.float32
                            ),
                            "next.done": np.array(
                                [bool(frame["next.done"])], dtype=bool
                            ),
                            "next.success": np.array(
                                [bool(frame["next.success"])], dtype=bool
                            ),
                        }
                    )

                dst.save_episode()
                pbar.update(1)

    dst.finalize()
    print(
        f"\nDone.\n"
        f"  Saved to       : {output_root}\n"
        f"  Total episodes : {dst.num_episodes}\n"
        f"  Total frames   : {dst.num_frames}\n"
    )
    return dst


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys

    if "--create-dataset" in _sys.argv:
        create_augmented_dataset(
            source_repo_id="lerobot/pusht",
            output_repo_id="local/pusht_augmented_v2",
            output_root="outputs/datasets/pusht_augmented_v2",
            n_augments=2,
            max_angle_deg=180.0,
            max_translate_coord=50.0,
            include_original=True,
            seed=42,
        )
    else:
        dataset = LeRobotDataset("lerobot/pusht")
        augmentations = [
            (PushTAugmentation(angle_deg=30), "Rotate +30° CW"),
            (PushTAugmentation(angle_deg=-30), "Rotate -30° CCW"),
            (PushTAugmentation(tx_coord=50), "Translate +50 right"),
            (PushTAugmentation(ty_coord=50), "Translate +50 down"),
            (
                PushTAugmentation(angle_deg=20, tx_coord=30, ty_coord=-20),
                "Rotate 20° + Translate",
            ),
        ]
        visualize_augmentations(
            dataset,
            augmentations=augmentations,
            n_frames=4,
            episode_idx=0,
            save_path="pusht_augmentation.png",
        )
