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

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torch.utils.data import default_collate
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 96
COORD_SIZE = 512
SCALE = IMG_SIZE / COORD_SIZE  # 0.1875

CX_IMG, CY_IMG = IMG_SIZE / 2, IMG_SIZE / 2  # (48, 48)
CX_COORD, CY_COORD = COORD_SIZE / 2, COORD_SIZE / 2  # (256, 256)


# ─── Core geometric transforms ────────────────────────────────────────────────


def _rotate_image(img: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate image (C, H, W) float32 [0,1] by angle_deg degrees around center.

    Uses TF.affine (same as all other image ops) so the angle convention is
    consistent: positive angle = clockwise when viewed on screen.

    Pad with white before rotation so bilinear interpolation at the boundary
    samples real white pixels instead of repeating the edge, avoiding dark border
    artifacts. Crop back to the original size after rotation.
    """
    h, w = img.shape[-2], img.shape[-1]
    # Padding needs to cover the furthest corner travel during rotation.
    # For a 96x96 image the corner is ~68px from center; ~25px extra is safe up to 45°.
    pad = int(np.ceil(max(h, w) * 0.3))
    padded = TF.pad(img, padding=pad, fill=1.0)
    rotated = TF.affine(
        padded,
        angle=angle_deg,
        translate=[0, 0],
        scale=1.0,
        shear=0,
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=1.0,
    )
    return TF.center_crop(rotated, [h, w])


def _translate_image(img: torch.Tensor, tx: float, ty: float) -> torch.Tensor:
    """Translate image by (tx, ty) pixels. tx>0 → right, ty>0 → down.

    Pad with white so bilinear interpolation at the entry edge samples real
    white pixels, then crop back, preserving the net translation offset.
    """
    h, w = img.shape[-2], img.shape[-1]
    # Extra margin of 2 pixels beyond the translation distance is enough.
    pad = int(np.ceil(max(abs(tx), abs(ty)))) + 2
    padded = TF.pad(img, padding=pad, fill=1.0)
    shifted = TF.affine(
        padded,
        angle=0,
        translate=[tx, ty],
        scale=1.0,
        shear=0,
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=1.0,
    )
    # Crop the padded region back out, keeping only the translated content.
    return TF.center_crop(shifted, [h, w])


def _rotate_coords(xy: torch.Tensor, angle_deg: float) -> torch.Tensor:
    theta = np.radians(angle_deg)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    dx = xy[..., 0] - CX_COORD
    dy = xy[..., 1] - CY_COORD

    # ⭐ 改成 CW（和 TF.affine 一致）
    x_new = CX_COORD + dx * cos_t - dy * sin_t
    y_new = CY_COORD + dx * sin_t + dy * cos_t

    return torch.stack([x_new, y_new], dim=-1)


def _translate_coords(xy: torch.Tensor, tx_img: float, ty_img: float) -> torch.Tensor:
    """
    Translate 2-D coordinates by (tx_img, ty_img) image pixels,
    converted to coordinate space.
    """
    dx_coord = tx_img / SCALE
    dy_coord = ty_img / SCALE
    offset = torch.tensor([dx_coord, dy_coord], dtype=xy.dtype)
    return xy + offset


def _is_valid_coords(xy: torch.Tensor) -> torch.Tensor:
    """
    Check if coords are within [0, COORD_SIZE].
    Supports (..., 2) shape.
    """
    return (
        (xy[..., 0] >= 0)
        & (xy[..., 0] <= COORD_SIZE)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] <= COORD_SIZE)
    )


# ─── Augmentation class ───────────────────────────────────────────────────────


class PushTAugmentation:
    """
    对 PushT 单帧 dict 做一致的几何增强：
      - 旋转：对图像与 state/action 坐标同步旋转
      - 平移：对图像与 state/action 坐标同步平移
    两种操作可叠加，顺序为：先旋转，后平移。
    """

    def __init__(
        self, angle_deg: float = 0.0, tx_img: float = 0.0, ty_img: float = 0.0
    ):
        """
        Args:
            angle_deg : 旋转角度（度），正值 = 逆时针（CCW）
            tx_img    : 水平平移（图像像素），正值 = 向右
            ty_img    : 垂直平移（图像像素），正值 = 向下
        """
        self.angle_deg = angle_deg
        self.tx_img = tx_img
        self.ty_img = ty_img

    def __call__(self, frame: dict) -> dict:
        aug = dict(frame)

        # ── Image ───────────────────────────────────────────
        img = frame["observation.image"]  # (3, 96, 96), float32 [0,1]
        if self.angle_deg != 0.0:
            img = _rotate_image(img, self.angle_deg)
        if self.tx_img != 0.0 or self.ty_img != 0.0:
            img = _translate_image(img, self.tx_img, self.ty_img)
        aug["observation.image"] = img

        # ── State ───────────────────────────────────────────
        state = frame["observation.state"].clone()  # (2,)
        if self.angle_deg != 0.0:
            state = _rotate_coords(state, self.angle_deg)
        if self.tx_img != 0.0 or self.ty_img != 0.0:
            state = _translate_coords(state, self.tx_img, self.ty_img)
        aug["observation.state"] = state

        # ── Action ──────────────────────────────────────────
        action = frame["action"].clone()  # (2,)
        if self.angle_deg != 0.0:
            action = _rotate_coords(action, self.angle_deg)
        if self.tx_img != 0.0 or self.ty_img != 0.0:
            action = _translate_coords(action, self.tx_img, self.ty_img)
        aug["action"] = action

        return aug


# ─── Visualization ────────────────────────────────────────────────────────────


def _draw_frame(ax, frame: dict, title: str):
    """在 ax 上绘制图像，并叠加 state（绿点）→ action（红箭头+红三角）。"""
    img = frame["observation.image"].permute(1, 2, 0).numpy()  # (H, W, 3)
    img = np.clip(img, 0.0, 1.0)
    ax.imshow(img, origin="upper")

    # 坐标空间 → 图像像素
    state_px = frame["observation.state"].numpy() * SCALE  # (2,)
    action_px = frame["action"].numpy() * SCALE  # (2,)

    ax.plot(state_px[0], state_px[1], "go", markersize=7, label="state", zorder=5)
    ax.annotate(
        "",
        xy=action_px,
        xytext=state_px,
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        zorder=6,
    )
    ax.plot(action_px[0], action_px[1], "r^", markersize=7, label="action", zorder=5)
    ax.set_title(title, fontsize=8, pad=3)
    ax.axis("off")


def visualize_augmentations(
    dataset,
    augmentations: list[tuple],
    n_frames: int = 4,
    episode_idx: int = 0,
    save_path: str = "pusht_augmentation.png",
):
    """
    生成可视化对比图：左列为原始帧，右侧各列为不同增强结果。

    Args:
        augmentations : list of (PushTAugmentation, label_str)
        n_frames      : 从 episode 中均匀采样的帧数
        episode_idx   : 使用哪个 episode
        save_path     : 输出图片路径
    """
    ep = dataset.meta.episodes[episode_idx]
    from_idx = ep["dataset_from_index"]
    to_idx = ep["dataset_to_index"]
    ep_len = to_idx - from_idx

    frame_indices = np.linspace(0, ep_len - 1, n_frames, dtype=int) + from_idx

    n_cols = 1 + len(augmentations)
    fig, axes = plt.subplots(
        n_frames,
        n_cols,
        figsize=(3.0 * n_cols, 3.2 * n_frames),
        squeeze=False,
    )

    for row, fi in enumerate(frame_indices):
        frame = dataset[int(fi)]
        ts = float(frame["timestamp"])

        _draw_frame(axes[row, 0], frame, f"Original  t={ts:.2f}s")
        for col, (aug_fn, label) in enumerate(augmentations, start=1):
            aug_frame = aug_fn(frame)
            _draw_frame(axes[row, col], aug_frame, label)

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
        f"PushT Geometric Augmentation  (episode {episode_idx})",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"[OK] 可视化已保存 → {save_path}")
    plt.close()


# ─── Demo ─────────────────────────────────────────────────────────────────────

# ─── Batch-level augmentation (for DataLoader collate_fn) ─────────────────────


def apply_equivariant_aug(
    batch: dict,
    max_angle_deg: float = 180.0,
    max_translate_coord: float = 150.0,
    arena_size: float = float(COORD_SIZE),
) -> dict:
    """
    Equivariant rigid-body augmentation applied to a collated batch.

    Applies the SAME random rotation + translation to every modality:
      - observation.image  (B, T_obs, C, H, W)  float32 [0, 1]
      - observation.state  (B, T_obs, 2)         agent_pos in [0, 512]
      - action             (B, T_act, 2)         target_pos in [0, 512]

    One independent rigid transform is sampled per batch element.
    Rotation and translation are fused into a single TF.affine call with
    white-pad + center-crop to avoid dark border artifacts.

    Must be called BEFORE the preprocessor (which normalises values).

    Args:
        batch               : dict returned by DataLoader (already collated)
        max_angle_deg       : uniform sample range for rotation [-max, +max]
        max_translate_coord : max translation in coordinate space (0-512 units)
        arena_size          : coordinate space size (default 512)
    """
    imgs = batch["observation.image"]  # (B, T_obs, C, H, W)
    B, T_obs, C, H, W = imgs.shape

    # ── Sample one rigid transform per batch item ──────────────────────────
    angles_deg = torch.zeros(B)
    tx_coord = torch.zeros(B)
    ty_coord = torch.zeros(B)

    MAX_TRY = 10

    for b in range(B):
        coords_state = batch["observation.state"][b]  # (T,2)
        coords_action = batch["action"][b]

        for _ in range(MAX_TRY):
            angle = float(torch.empty(1).uniform_(-max_angle_deg, max_angle_deg))
            tx = float(
                torch.empty(1).uniform_(-max_translate_coord, max_translate_coord)
            )
            ty = float(
                torch.empty(1).uniform_(-max_translate_coord, max_translate_coord)
            )

            theta = math.radians(angle)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            def transform(xy):
                cx = xy[..., 0] - 256.0
                cy = xy[..., 1] - 256.0
                x_new = cx * cos_t - cy * sin_t + 256.0 + tx
                y_new = cx * sin_t + cy * cos_t + 256.0 + ty
                return torch.stack([x_new, y_new], dim=-1)

            new_state = transform(coords_state)
            new_action = transform(coords_action)

            if _is_valid_coords(new_state).all() and _is_valid_coords(new_action).all():
                angles_deg[b] = angle
                tx_coord[b] = tx
                ty_coord[b] = ty
                break
        else:
            # fallback（极少发生）
            angles_deg[b] = 0.0
            tx_coord[b] = 0.0
            ty_coord[b] = 0.0

    # Coordinate-space translation → image-pixel translation
    tx_px = (tx_coord * H / arena_size).tolist()
    ty_px = (ty_coord * H / arena_size).tolist()

    # ── Augment images ─────────────────────────────────────────────────────
    # Combine rotation + translation in one TF.affine call.
    # Pad with white first so bilinear interpolation at the boundary never
    # samples the "repeat-edge" artifact that creates the dark border line.
    pad = int(math.ceil(max(H, W) * 0.3))  # ~29px for 96×96, safe up to ±45°
    imgs_aug = torch.empty_like(imgs)
    for b in range(B):
        for t in range(T_obs):
            frame = imgs[b, t]  # (C, H, W)
            padded = TF.pad(frame, padding=pad, fill=1.0)
            augmented = TF.affine(
                padded,
                angle=angles_deg[b].item(),
                translate=[tx_px[b], ty_px[b]],
                scale=1.0,
                shear=0,
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=1.0,
            )
            imgs_aug[b, t] = TF.center_crop(augmented, [H, W])
    batch["observation.image"] = imgs_aug

    # ── Augment 2-D coordinates ────────────────────────────────────────────
    # TF.affine positive angle = CW rotation. In image space (y-axis DOWN)
    # the matching CW coord transform is:
    #   x' = (x-256)*cos - (y-256)*sin + 256 + tx_coord
    #   y' = (x-256)*sin + (y-256)*cos + 256 + ty_coord
    dev = batch["observation.state"].device
    theta = angles_deg * (math.pi / 180.0)
    cos_t = theta.cos().to(dev)[:, None]  # (B, 1)  broadcastable over T
    sin_t = theta.sin().to(dev)[:, None]
    tx_c = tx_coord.to(dev)[:, None]
    ty_c = ty_coord.to(dev)[:, None]

    def _transform_coords(coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, T, 2)
        cx = coords[..., 0] - 256.0
        cy = coords[..., 1] - 256.0
        x_new = cx * cos_t - cy * sin_t + 256.0 + tx_c
        y_new = cx * sin_t + cy * cos_t + 256.0 + ty_c
        return torch.stack([x_new, y_new], dim=-1)

    batch["observation.state"] = _transform_coords(batch["observation.state"])
    batch["action"] = _transform_coords(batch["action"])
    return batch


class AugmentCollate:
    """
    A picklable collate_fn that assembles samples into a batch and immediately
    applies equivariant geometric augmentation (rotation + translation).

    Picklability is required when num_workers > 0 so worker processes can
    receive this object via pickle. Using a top-level class (instead of a
    closure) guarantees this.

    Usage:
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            sampler=sampler,
            collate_fn=AugmentCollate(max_angle_deg=180, max_translate_coord=150),
            num_workers=4,
        )
    """

    def __init__(
        self,
        max_angle_deg: float = 180.0,
        max_translate_coord: float = 150.0,
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


# ─── Dataset-level augmentation: write a new LeRobotDataset ──────────────────


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
    """
    Create a new LeRobotDataset by augmenting an existing one.

    For every episode in the source, one or more augmented copies are written
    with the SAME random rigid transform (rotation + translation) applied to
    every frame of that episode, keeping the episode spatially consistent.

    Augmented modalities (same transform applied to all three):
      - observation.image : pad → TF.affine → center_crop (white fill, no border line)
      - observation.state : 2-D coordinate rotation + translation
      - action            : 2-D coordinate rotation + translation

    Other scalar fields (next.reward, next.done, next.success) are copied as-is
    because reward is determined by relative positions, which are invariant to
    rigid transforms of the whole scene.

    Args:
        source_repo_id      : HF or local dataset to augment
        output_repo_id      : repo_id label for the new dataset (local name)
        output_root         : directory where the new dataset is saved on disk
        n_augments          : number of augmented copies per source episode
        max_angle_deg       : uniform sample range for rotation  [-max, +max] degrees
        max_translate_coord : uniform sample range for translation [-max, +max] coord units
        include_original    : also include the original unaugmented episodes
        seed                : numpy random seed for reproducibility

    Returns:
        The newly created LeRobotDataset.
    """
    rng = np.random.default_rng(seed)

    print(f"Loading source dataset: {source_repo_id}")
    src = LeRobotDataset(source_repo_id)

    # Features for the new dataset (strip DEFAULT_FEATURES which are auto-managed)
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
        image_writer_threads=4,  # parallel PNG writing → faster
        vcodec="libsvtav1",  # same codec as source
    )

    # Whether each pass is augmented (None = original)
    passes = (["original"] if include_original else []) + [
        f"aug_{i}" for i in range(n_augments)
    ]
    total_episodes = src.num_episodes * len(passes)

    print(
        f"\nSource episodes : {src.num_episodes}\n"
        f"Passes per ep   : {len(passes)}  ({passes})\n"
        f"Total new eps   : {total_episodes}\n"
    )

    H, W = IMG_SIZE, IMG_SIZE
    pad = int(np.ceil(max(H, W) * 0.3))  # ~29px for 96×96

    with tqdm(total=total_episodes, desc="Writing episodes", unit="ep") as pbar:
        for ep_idx in range(src.num_episodes):
            ep = src.meta.episodes[ep_idx]
            from_idx = int(ep["dataset_from_index"])
            to_idx = int(ep["dataset_to_index"])
            task = ep["tasks"][0]  # task string (list with one entry per episode)

            for pass_name in passes:
                # ── Sample rigid transform for this episode ────────────────
                if pass_name == "original":
                    angle_deg = 0.0
                    tx_coord = 0.0
                    ty_coord = 0.0
                else:
                    MAX_TRY = 20
                    for _ in range(MAX_TRY):
                        angle_deg = float(rng.uniform(-max_angle_deg, max_angle_deg))
                        tx_coord = float(
                            rng.uniform(-max_translate_coord, max_translate_coord)
                        )
                        ty_coord = float(
                            rng.uniform(-max_translate_coord, max_translate_coord)
                        )

                        cos_t = math.cos(math.radians(angle_deg))
                        sin_t = math.sin(math.radians(angle_deg))

                        def _test_coord(xy):
                            dx, dy = float(xy[0]) - 256.0, float(xy[1]) - 256.0
                            x_new = dx * cos_t - dy * sin_t + 256.0 + tx_coord
                            y_new = dx * sin_t + dy * cos_t + 256.0 + ty_coord
                            return np.array([x_new, y_new])

                        valid = True
                        for fi in range(from_idx, to_idx):
                            frame = src[fi]
                            s = _test_coord(frame["observation.state"].numpy())
                            a = _test_coord(frame["action"].numpy())

                            if not (
                                (0 <= s[0] <= 512 and 0 <= s[1] <= 512)
                                and (0 <= a[0] <= 512 and 0 <= a[1] <= 512)
                            ):
                                valid = False
                                break

                        if valid:
                            break
                    else:
                        # fallback
                        angle_deg = 0.0
                        tx_coord = 0.0
                        ty_coord = 0.0

                cos_t = math.cos(math.radians(angle_deg))
                sin_t = math.sin(math.radians(angle_deg))
                tx_px = tx_coord * SCALE  # coord → image pixels
                ty_px = ty_coord * SCALE

                # ── Write every frame of this episode ──────────────────────
                for fi in range(from_idx, to_idx):
                    frame = src[fi]

                    # Image: (3, H, W) float32 → augment → (H, W, 3) uint8
                    img = frame["observation.image"]
                    if pass_name != "original":
                        padded = TF.pad(img, padding=pad, fill=1.0)
                        img = TF.affine(
                            padded,
                            angle=angle_deg,
                            translate=[tx_px, ty_px],
                            scale=1.0,
                            shear=0,
                            interpolation=TF.InterpolationMode.BILINEAR,
                            fill=1.0,
                        )
                        img = TF.center_crop(img, [H, W])
                    img_np = (
                        (img.permute(1, 2, 0) * 255)
                        .clamp(0, 255)
                        .to(torch.uint8)
                        .numpy()
                    )

                    # Coordinates: rotate then translate in coord space
                    def _aug_coord(xy: np.ndarray) -> np.ndarray:
                        if pass_name == "original":
                            return xy.astype(np.float32)
                        dx, dy = float(xy[0]) - 256.0, float(xy[1]) - 256.0
                        return np.array(
                            [
                                dx * cos_t - dy * sin_t + 256.0 + tx_coord,
                                dx * sin_t + dy * cos_t + 256.0 + ty_coord,
                            ],
                            dtype=np.float32,
                        )

                    state = _aug_coord(frame["observation.state"].numpy())
                    action = _aug_coord(frame["action"].numpy())

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


if __name__ == "__main__":
    import sys as _sys

    if "--create-dataset" in _sys.argv:
        # ── Build augmented dataset ────────────────────────────────────────
        # Usage: python pusht_augment.py --create-dataset
        create_augmented_dataset(
            source_repo_id="lerobot/pusht",
            output_repo_id="local/pusht_augmented_v2",
            output_root="outputs/datasets/pusht_augmented_v2",
            n_augments=2,  # 2 extra augmented copies per episode
            max_angle_deg=180.0,  # full rotation range
            max_translate_coord=50.0,  # ±100 in [0,512] coord space
            include_original=True,  # keep originals too
            seed=42,
        )
    else:
        # ── Visualise augmentation on a few frames ─────────────────────────
        dataset = LeRobotDataset("lerobot/pusht")

        augmentations = [
            (PushTAugmentation(angle_deg=30), "Rotate +30 deg CCW"),
            (PushTAugmentation(angle_deg=-30), "Rotate -30 deg CW"),
            (PushTAugmentation(tx_img=10, ty_img=0), "Translate +10px right"),
            (PushTAugmentation(tx_img=0, ty_img=10), "Translate +10px down"),
            (
                PushTAugmentation(angle_deg=20, tx_img=8, ty_img=-5),
                "Rotate 20deg + Translate",
            ),
        ]

        visualize_augmentations(
            dataset,
            augmentations=augmentations,
            n_frames=4,
            episode_idx=0,
            save_path="pusht_augmentation.png",
        )
