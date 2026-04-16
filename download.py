from pathlib import Path
from huggingface_hub import snapshot_download


def download_from_hf(
    repo_id: str,
    local_dir: str,
    repo_type: str = "model",  # "model" 或 "dataset"
    revision: str | None = None,
    token: str | None = None,
):
    """
    repo_id: 例如 "lerobot/diffusion_pusht" 或 "username/dataset_name"
    local_dir: 本地目标目录
    repo_type: "model" / "dataset"
    revision: 分支、tag 或 commit hash（可选）
    token: 私有仓库访问 token（可选）
    """
    local_dir = str(Path(local_dir).expanduser().resolve())
    path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # 直接落地文件
        revision=revision,
        token=token,
    )
    print(f"下载完成: {repo_id} -> {path}")


if __name__ == "__main__":
    # 例1：下载模型权重
    download_from_hf(
        repo_id="lerobot/diffusion_pusht",
        repo_type="model",
        local_dir="~/Documents/Diffusion/data/lerobot/diffusion_pusht",
    )

    # 例2：下载数据集
    # download_from_hf(
    #     repo_id="lerobot/pusht",
    #     repo_type="dataset",
    #     local_dir="~/hf_downloads/pusht_dataset",
    # )
