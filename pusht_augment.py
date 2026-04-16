import math
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import default_collate
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
IMG_SIZE = 96
COORD_SIZE = 512
SCALE = IMG_SIZE / COORD_SIZE

CX = COORD_SIZE / 2  # 256


# ─────────────────────────────────────────────────────────────
# Core: Single Source of Truth
# ─────────────────────────────────────────────────────────────
class RigidTransform:
    def __init__(self, angle_deg: float, tx: float, ty: float):
        theta = math.radians(angle_deg)
        self.angle_deg = angle_deg
        self.cos_t = math.cos(theta)
        self.sin_t = math.sin(theta)
        self.tx = tx
        self.ty = ty

    def apply_coords(self, xy):
        cx = xy[..., 0] - CX
        cy = xy[..., 1] - CX

        x_new = CX + cx * self.cos_t - cy * self.sin_t + self.tx
        y_new = CX + cx * self.sin_t + cy * self.cos_t + self.ty

        if isinstance(xy, torch.Tensor):
            return torch.stack([x_new, y_new], dim=-1)
        else:
            return np.stack([x_new, y_new], axis=-1)

    def is_valid(self, coords):
        return (
            (coords[..., 0] >= 0)
            & (coords[..., 0] <= COORD_SIZE)
            & (coords[..., 1] >= 0)
            & (coords[..., 1] <= COORD_SIZE)
        )


# ─────────────────────────────────────────────────────────────
# Image Transform
# ─────────────────────────────────────────────────────────────
def apply_image_transform(img: torch.Tensor, tfm: RigidTransform):
    H, W = img.shape[-2:]
    pad = int(np.ceil(max(H, W) * 0.3))

    padded = TF.pad(img, padding=pad, fill=1.0)

    tx_px = tfm.tx * (H / COORD_SIZE)
    ty_px = tfm.ty * (H / COORD_SIZE)

    out = TF.affine(
        padded,
        angle=tfm.angle_deg,
        translate=[tx_px, ty_px],
        scale=1.0,
        shear=0,
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=1.0,
    )

    return TF.center_crop(out, [H, W])


# ─────────────────────────────────────────────────────────────
# Resample Transform
# ─────────────────────────────────────────────────────────────
def sample_valid_transform(
    state,
    action,
    max_angle=180,
    max_trans=100,
    max_try=20,
):
    for _ in range(max_try):
        angle = float(np.random.uniform(-max_angle, max_angle))
        tx = float(np.random.uniform(-max_trans, max_trans))
        ty = float(np.random.uniform(-max_trans, max_trans))

        tfm = RigidTransform(angle, tx, ty)

        s_new = tfm.apply_coords(state)
        a_new = tfm.apply_coords(action)

        if tfm.is_valid(s_new).all() and tfm.is_valid(a_new).all():
            return tfm

    return None  # resample失败


# ─────────────────────────────────────────────────────────────
# Batch-level augmentation
# ─────────────────────────────────────────────────────────────
def apply_batch_aug(batch, max_angle=180, max_trans=100):
    B = batch["observation.state"].shape[0]

    imgs_out = []
    states_out = []
    actions_out = []

    for b in range(B):
        state = batch["observation.state"][b]
        action = batch["action"][b]

        tfm = sample_valid_transform(state, action, max_angle, max_trans)

        if tfm is None:
            tfm = RigidTransform(0, 0, 0)

        imgs = batch["observation.image"][b]

        imgs_aug = torch.stack([apply_image_transform(img, tfm) for img in imgs])

        imgs_out.append(imgs_aug)
        states_out.append(tfm.apply_coords(state))
        actions_out.append(tfm.apply_coords(action))

    batch["observation.image"] = torch.stack(imgs_out)
    batch["observation.state"] = torch.stack(states_out)
    batch["action"] = torch.stack(actions_out)

    return batch


class AugmentCollate:
    def __init__(self, max_angle=180, max_trans=100):
        self.max_angle = max_angle
        self.max_trans = max_trans

    def __call__(self, samples):
        batch = default_collate(samples)
        return apply_batch_aug(batch, self.max_angle, self.max_trans)


# ─────────────────────────────────────────────────────────────
# Dataset-level augmentation
# ─────────────────────────────────────────────────────────────
def create_augmented_dataset(
    source_repo_id="lerobot/pusht",
    output_repo_id="local/pusht_aug",
    output_root="outputs/pusht_aug",
    n_augments=2,
    max_angle=180,
    max_trans=100,
    include_original=True,
    seed=42,
):
    rng = np.random.default_rng(seed)

    src = LeRobotDataset(source_repo_id)

    user_features = {k: v for k, v in src.features.items() if k not in DEFAULT_FEATURES}

    output_root = Path(output_root)
    if output_root.exists():
        import shutil

        shutil.rmtree(output_root)

    dst = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=src.fps,
        features=user_features,
        root=output_root,
        use_videos=True,
    )

    passes = (["original"] if include_original else []) + [
        f"aug_{i}" for i in range(n_augments)
    ]

    for ep_idx in tqdm(range(src.num_episodes)):
        ep = src.meta.episodes[ep_idx]
        from_idx = ep["dataset_from_index"]
        to_idx = ep["dataset_to_index"]

        for p in passes:

            if p == "original":
                tfm = RigidTransform(0, 0, 0)
            else:
                tfm = None

                for _ in range(20):
                    candidate = RigidTransform(
                        rng.uniform(-max_angle, max_angle),
                        rng.uniform(-max_trans, max_trans),
                        rng.uniform(-max_trans, max_trans),
                    )

                    valid = True
                    for fi in range(from_idx, to_idx):
                        frame = src[fi]

                        s = candidate.apply_coords(frame["observation.state"].numpy())
                        a = candidate.apply_coords(frame["action"].numpy())

                        if not (candidate.is_valid(s) and candidate.is_valid(a)):
                            valid = False
                            break

                    if valid:
                        tfm = candidate
                        break

                if tfm is None:
                    tfm = RigidTransform(0, 0, 0)

            # write episode
            for fi in range(from_idx, to_idx):
                frame = src[fi]

                img = apply_image_transform(frame["observation.image"], tfm)
                img_np = (
                    (img.permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).numpy()
                )

                state = tfm.apply_coords(frame["observation.state"].numpy())
                action = tfm.apply_coords(frame["action"].numpy())

                dst.add_frame(
                    {
                        "task": ep["tasks"][0],
                        "observation.image": img_np,
                        "observation.state": state.astype(np.float32),
                        "action": action.astype(np.float32),
                        "next.reward": np.array(
                            [frame["next.reward"]], dtype=np.float32
                        ),
                        "next.done": np.array([frame["next.done"]], dtype=bool),
                        "next.success": np.array([frame["next.success"]], dtype=bool),
                    }
                )

            dst.save_episode()

    dst.finalize()
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
            max_angle=180.0,  # full rotation range
            max_trans=50.0,  # ±100 in [0,512] coord space
            include_original=True,  # keep originals too
            seed=42,
        )
    else:
        exit("Please specify --create-dataset to build the augmented dataset.")
