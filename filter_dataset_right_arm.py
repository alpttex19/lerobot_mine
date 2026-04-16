"""
过滤数据集，只保留右臂关节位置 (8维) + right_color + chest_camera
输入: /home/party/Documents/data/lerobot/placemouse_datasets
输出: /home/party/Documents/data/lerobot/placemouse_right_arm
"""

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ── 路径配置 ──────────────────────────────────────────────────────────────────
SRC = Path("/home/party/Documents/data/lerobot/placemouse_datasets")
DST = Path("/home/party/Documents/data/lerobot/placemouse_right_arm")

# 保留的 action / state 维度索引（右臂关节位置）
KEEP_INDICES = list(range(8, 16))  # 8~15
KEEP_NAMES = [
    "right_joint1_position",
    "right_joint2_position",
    "right_joint3_position",
    "right_joint4_position",
    "right_joint5_position",
    "right_joint6_position",
    "right_joint7_position",
    "right_gripper_position",
]

# 保留的相机 key
KEEP_CAMERAS = {"right_color", "chest_camera"}
REMOVE_CAMERAS = {"left_color"}  # 仅记录，不复制视频


# ── 1. 处理 Parquet 文件 ──────────────────────────────────────────────────────
def process_parquet(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(src_path)

    new_columns = {}
    for col_name in table.schema.names:
        col = table[col_name]
        if col_name in ("action", "observation.state"):
            # 切片每行，只保留 KEEP_INDICES
            sliced = pa.array(
                [
                    [row.as_py()[i] for i in KEEP_INDICES]
                    for row in col
                ],
                type=pa.list_(pa.float32()),
            )
            new_columns[col_name] = sliced
        else:
            new_columns[col_name] = col

    new_table = pa.table(new_columns)
    pq.write_table(new_table, dst_path)


def process_all_parquets() -> None:
    src_data = SRC / "data"
    dst_data = DST / "data"
    parquet_files = sorted(src_data.rglob("*.parquet"))
    print(f"处理 {len(parquet_files)} 个 parquet 文件...")
    for i, src_file in enumerate(parquet_files):
        rel = src_file.relative_to(SRC)
        dst_file = DST / rel
        process_parquet(src_file, dst_file)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(parquet_files)}")
    print("Parquet 处理完成")


# ── 2. 复制视频（只复制保留的相机）──────────────────────────────────────────
def copy_videos() -> None:
    src_videos = SRC / "videos"
    dst_videos = DST / "videos"
    if not src_videos.exists():
        return

    for camera_dir in src_videos.iterdir():
        # camera_dir 名称形如 observation.images.right_color
        camera_key = camera_dir.name.split(".")[-1]  # 取最后一段
        if camera_key not in KEEP_CAMERAS:
            print(f"跳过相机: {camera_dir.name}")
            continue
        dst_camera = dst_videos / camera_dir.name
        print(f"复制视频: {camera_dir.name} ...")
        shutil.copytree(camera_dir, dst_camera)
    print("视频复制完成")


# ── 3. 复制 meta（episodes, tasks.parquet）──────────────────────────────────
def copy_meta() -> None:
    dst_meta = DST / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    # tasks.parquet 直接复制
    src_tasks = SRC / "meta" / "tasks.parquet"
    if src_tasks.exists():
        shutil.copy2(src_tasks, dst_meta / "tasks.parquet")

    # episodes 目录直接复制
    src_episodes = SRC / "meta" / "episodes"
    if src_episodes.exists():
        shutil.copytree(src_episodes, dst_meta / "episodes")


# ── 4. 生成新的 info.json ──────────────────────────────────────────────────────
def write_info_json() -> None:
    with open(SRC / "meta" / "info.json") as f:
        info = json.load(f)

    features = info["features"]

    # 更新 action
    features["action"]["shape"] = [len(KEEP_NAMES)]
    features["action"]["names"] = KEEP_NAMES

    # 更新 observation.state
    features["observation.state"]["shape"] = [len(KEEP_NAMES)]
    features["observation.state"]["names"] = KEEP_NAMES

    # 移除不需要的相机
    for cam in list(REMOVE_CAMERAS):
        key = f"observation.images.{cam}"
        if key in features:
            del features[key]
            print(f"info.json: 移除 {key}")

    with open(DST / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    print("info.json 已更新")


# ── 5. 生成新的 stats.json ────────────────────────────────────────────────────
def write_stats_json() -> None:
    with open(SRC / "meta" / "stats.json") as f:
        stats = json.load(f)

    new_stats = {}

    for key, val in stats.items():
        if key in ("action", "observation.state"):
            # 只保留 KEEP_INDICES 对应的统计值
            new_val = {}
            for stat_name, arr in val.items():
                if stat_name == "count":
                    new_val["count"] = arr  # count 不需要切片
                else:
                    new_val[stat_name] = [arr[i] for i in KEEP_INDICES]
            new_stats[key] = new_val

        elif key.startswith("observation.images."):
            cam_key = key.split(".")[-1]
            if cam_key in KEEP_CAMERAS:
                new_stats[key] = val
            else:
                print(f"stats.json: 移除 {key}")

        else:
            # timestamp, frame_index, episode_index, index, task_index 等直接保留
            new_stats[key] = val

    with open(DST / "meta" / "stats.json", "w") as f:
        json.dump(new_stats, f, indent=4)
    print("stats.json 已更新")


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main() -> None:
    if DST.exists():
        print(f"目标目录已存在: {DST}")
        ans = input("是否覆盖? [y/N] ").strip().lower()
        if ans != "y":
            print("已取消")
            return
        shutil.rmtree(DST)

    DST.mkdir(parents=True)

    process_all_parquets()
    copy_videos()
    copy_meta()
    write_info_json()
    write_stats_json()

    print(f"\n完成！新数据集路径: {DST}")
    print(f"Action/State 维度: {len(KEEP_NAMES)} (原来 62)")
    print(f"保留相机: {KEEP_CAMERAS}")


if __name__ == "__main__":
    main()
