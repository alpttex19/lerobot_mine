import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gym_pusht

# ================== 请确保这里的类与之前一致 ==================
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.utils import preprocess_observation


# 假设你已经有了 RandomGoalWrapper 和 FeatureExtractor
from test_feature import RandomGoalWrapper, FeatureExtractor, collect_data_for_probe


# =================================================================
def visualize_linear_probe(
    pretrained_path, probe_model, n_episodes=2, save_dir="probe_visualizations"
):
    """运行环境，使用训练好的探针权重可视化预测位姿"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载策略和 Hook
    policy = DiffusionPolicy.from_pretrained(pretrained_path)
    policy.eval()
    policy.to(device)
    target_layer_name = "diffusion.rgb_encoder.backbone.7.1.conv2"  # 替换为你的目标层
    extractor = FeatureExtractor(policy, target_layer_name)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=pretrained_path,
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )
    env = RandomGoalWrapper(
        gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            observation_width=96,
            observation_height=96,
            visualization_width=384,
            visualization_height=384,
        )  # 渲染大一点好看
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # 提取线性回归的权重和偏置
    W = probe_model.coef_  # shape: (8, 512)
    b = probe_model.intercept_  # shape: (8,)
    print(f"\n=== 开始可视化，保存路径: {save_dir} ===")
    for ep in range(n_episodes):
        policy.reset()
        obs, info = env.reset(seed=100 + ep)
        done = False
        step_count = 0
        while not done and step_count < 200:
            # 1. 获取真实位姿 (用于绿线标注)
            block_pos = info["block_pose"][:2]
            block_angle = info["block_pose"][2]
            goal_pos = info["goal_pose"][:2]
            goal_angle = info["goal_pose"][2]
            # 2. 模型推理以触发 Hook
            obs_dict = preprocess_observation(obs)
            obs_dict_norm = preprocessor(obs_dict)
            with torch.inference_mode():
                action = policy.select_action(obs_dict_norm)
                if extractor.activations is not None:
                    feat = extractor.activations
                    pooled_feat = feat.mean(dim=(2, 3))
                    latent_vector = pooled_feat[-1].cpu().numpy().flatten()  # (512,)
                    # 3. 使用线性权重预测位姿: y = xW^T + b
                    pred_pose = np.dot(latent_vector, W.T) + b
                    # pred_pose 顺序: [Block_X, Block_Y, Block_Sin, Block_Cos, Goal_X, Goal_Y, Goal_Sin, Goal_Cos]
                    # 4. 逆变换三角函数为角度
                    pred_block_angle = np.arctan2(pred_pose[2], pred_pose[3])
                    pred_goal_angle = np.arctan2(pred_pose[6], pred_pose[7])
                    # 5. 每隔 10 帧可视化一次
                    if step_count % 10 == 0:
                        # 获取渲染图像 (H, W, C)
                        img = env.render()
                        # ⚠️ 注意坐标系：gym_pusht 的物理坐标与图像像素坐标的映射
                        # 通常物理 Y 轴朝上，图像 Y 轴朝下，需要翻转 Y
                        # PushT 默认物理空间大约在 X:[0, 384], Y:[0, 384]
                        # 渲染图大小为 384x384，需要计算缩放比例
                        scale = 384.0 / 512.0
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(img)
                        # 画真实位姿 (绿色)
                        draw_t(
                            ax,
                            block_pos[0] * scale,
                            block_pos[1] * scale,
                            block_angle,
                            color="lime",
                            label="True Block",
                        )
                        draw_t(
                            ax,
                            goal_pos[0] * scale,
                            goal_pos[1] * scale,
                            goal_angle,
                            color="lime",
                            linestyle="--",
                            label="True Goal",
                        )
                        # 画预测位姿 (红色)
                        draw_t(
                            ax,
                            pred_pose[0] * scale,
                            pred_pose[1] * scale,
                            pred_block_angle,
                            color="red",
                            label="Pred Block",
                        )
                        draw_t(
                            ax,
                            pred_pose[4] * scale,
                            pred_pose[5] * scale,
                            pred_goal_angle,
                            color="red",
                            linestyle="--",
                            label="Pred Goal",
                        )
                        ax.set_title(f"Episode {ep} | Step {step_count}")
                        ax.legend(loc="upper right")
                        save_path = Path(save_dir) / f"ep{ep}_step{step_count}.png"
                        plt.savefig(save_path, bbox_inches="tight")
                        plt.close()
                extractor.activations = None
            action = postprocessor(action)
            action_np = action.to("cpu").numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            step_count += 1
        print(f"Episode {ep} 可视化完成。")
    env.close()
    extractor.remove()


def draw_t(ax, x, y, angle, color="lime", length=30, linestyle="-", label=None):
    """在图像上画一个带方向的 T 形标记（简化为带箭头的点）"""
    # 画中心点
    ax.plot(x, y, "o", color=color, markersize=8)
    # 画方向线 (根据角度)
    dx = length * np.cos(angle)
    dy = length * np.sin(angle)  # 图像Y轴向下，所以sin取负
    ax.plot([x, x + dx], [y, y + dy], color=color, linewidth=3, linestyle=linestyle)


# ================== 主程序运行示例 ==================
if __name__ == "__main__":
    PRETRAINED_PATH = (
        "outputs/train/diffusion_pusht_augmented/checkpoints/last/pretrained_model"
    )
    # 1. 先收集数据训练探针 (复用你之前的代码)
    print("收集数据并训练探针...")
    X, y = collect_data_for_probe(PRETRAINED_PATH, n_episodes=20)
    # 训练线性探针 (目标是 sin/cos 格式)
    probe = LinearRegression()
    probe.fit(X, y)
    print("\n探针权重形状:", probe.coef_.shape)
    print("探针偏置形状:", probe.intercept_.shape)
    # 2. 运行可视化
    print("开始可视化推理结果...")
    visualize_linear_probe(PRETRAINED_PATH, probe_model=probe, n_episodes=2)
