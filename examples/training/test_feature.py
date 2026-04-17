import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# --- 复用之前的 Wrapper 和 FeatureExtractor ---
class RandomGoalWrapper(gym.Wrapper):
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # 注意：为了验证模型是否真的能提取位姿，必须让环境有多样性
        # 这里解除你之前的注释，让目标随机化，逼迫模型去"看"
        rng = self.env.unwrapped.np_random
        x = float(rng.integers(100, 412))
        y = float(rng.integers(100, 412))
        theta = float(rng.uniform(0, 2 * np.pi))
        self.env.unwrapped.goal_pose = np.array([x, y, theta], dtype=np.float64)
        info["goal_pose"] = self.env.unwrapped.goal_pose
        return obs, info


class FeatureExtractor:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.target_layer = self._get_module_by_name(model, target_layer_name)
        if self.target_layer is None:
            raise ValueError(f"未找到层: {target_layer_name}")
        self.handle = self.target_layer.register_forward_hook(self._hook_fn)

    def _get_module_by_name(self, model, name):
        components = name.split(".")
        curr = model
        for comp in components:
            if hasattr(curr, comp):
                curr = getattr(curr, comp)
            else:
                return None
        return curr

    def _hook_fn(self, module, input, output):
        self.activations = output.detach()

    def remove(self):
        self.handle.remove()


def collect_data_for_probe(pretrained_path, n_episodes=10):
    """运行环境，收集 (隐式向量, 真实位姿) 数据对，角度转为 sin/cos"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    import gym_pusht

    policy = DiffusionPolicy.from_pretrained(pretrained_path)
    policy.eval()
    policy.to(device)

    # ⚠️ 填入你之前找到的最后一层卷积名称
    target_layer_name = "diffusion.rgb_encoder.backbone.7.1.conv2"
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
        )
    )

    all_features = []
    all_poses = []

    print(f"开始收集数据，共 {n_episodes} 个 episode...")
    for ep in range(n_episodes):
        policy.reset()
        obs, info = env.reset(seed=ep)
        done = False
        step_count = 0

        while not done and step_count < 200:
            # 1. 获取环境的真实位姿，并将角度转换为 sin/cos
            block_pos = info["block_pose"][:2]
            block_angle = info["block_pose"][2]
            goal_pos = info["goal_pose"][:2]
            goal_angle = info["goal_pose"][2]

            # ✅ 核心修改：将 theta 转换为 sin(theta) 和 cos(theta)
            block_pose = np.array(
                [block_pos[0], block_pos[1], np.sin(block_angle), np.cos(block_angle)]
            )
            goal_pose = np.array(
                [goal_pos[0], goal_pos[1], np.sin(goal_angle), np.cos(goal_angle)]
            )
            true_pose = np.concatenate([block_pose, goal_pose])  # 现在长度变成了 8

            # 2. 模型前向推理以触发 Hook
            obs_dict = preprocess_observation(obs)
            obs_dict_norm = preprocessor(obs_dict)
            with torch.inference_mode():
                action = policy.select_action(obs_dict_norm)

                # 3. 提取特征并进行全局池化
                if extractor.activations is not None:
                    feat = extractor.activations
                    pooled_feat = feat.mean(dim=(2, 3))
                    latent_vector = pooled_feat[-1].cpu().numpy()

                    all_features.append(latent_vector)
                    all_poses.append(true_pose)

                extractor.activations = None

            action = postprocessor(action)
            action_np = action.to("cpu").numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            step_count += 1

        print(f"Episode {ep} 完成，收集了 {step_count} 个样本。")

    env.close()
    extractor.remove()

    X = np.array(all_features)
    y = np.array(all_poses)
    print(f"数据收集完毕！特征矩阵形状: {X.shape}, 位姿矩阵形状: {y.shape}")
    return X, y


def train_and_evaluate_probe_linear(X, y):
    """训练线性回归探针并评估 sin/cos 预测"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n=== 开始训练三角函数线性探针 ===")
    probe = LinearRegression()
    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_test)

    print("\n=== 探针评估结果 ===")
    # ✅ 标签更新为 sin 和 cos
    labels = [
        "Block_X",
        "Block_Y",
        "Block_Sin_Theta",
        "Block_Cos_Theta",
        "Goal_X",
        "Goal_Y",
        "Goal_Sin_Theta",
        "Goal_Cos_Theta",
    ]

    # 调整画布大小以容纳 8 个子图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Linear Probe: Predicting Trigonometric Angle Encodings", fontsize=16)

    for i, label in enumerate(labels):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        print(f"{label:<20}: MSE = {mse:.4f}, R² = {r2:.4f}")

        ax = axes[i // 4, i % 4]
        ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.3, s=10)
        # 画对角线
        min_val = min(y_test[:, i].min(), y_pred[:, i].min())
        max_val = max(y_test[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        ax.set_xlabel("True Value")
        ax.set_ylabel("Predicted Value")
        ax.set_title(f"{label} (R²={r2:.3f})")

    plt.tight_layout()
    plt.savefig("linear_probe_trig_results.png", dpi=150)
    print("\n可视化图表已保存至: linear_probe_trig_results.png")


def train_and_evaluate_probe(X, y):
    """训练带归一化和正则化的线性和非线性探针"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    labels = [
        "Block_X",
        "Block_Y",
        "Block_Sin",
        "Block_Cos",
        "Goal_X",
        "Goal_Y",
        "Goal_Sin",
        "Goal_Cos",
    ]

    # ------------------- 线性探针 (基准) -------------------
    print("\n=== 1. 训练线性探针 ===")
    linear_probe = LinearRegression()
    linear_probe.fit(X_train, y_train)
    y_pred_linear = linear_probe.predict(X_test)

    # ------------------- 非线性探针 (修复版 MLP) -------------------
    print("\n=== 2. 训练非线性 MLP 探针 (带归一化和正则化) ===")

    # ✅ 关键修复 1：构建 Pipeline，先对数据做 StandardScaler (均值变0，方差变1)
    # ✅ 关键修复 2：增加 alpha 参数 (L2 正则化)，惩罚大权重，防止死记硬背
    # ✅ 关键修复 3：缩小网络规模，从 (64, 32) 降到 (32, 16)，减少参数量
    # 由于 Pipeline 机制，我们需要用 fit_transform 手动处理 y
    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_scaled = scaler_x.transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.transform(y_train)

    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=1000,
        alpha=0.01,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )

    mlp.fit(X_train_scaled, y_train_scaled)

    # 预测并逆归一化回原始尺度，以便计算真实的 R²
    y_pred_mlp_scaled = mlp.predict(X_test_scaled)
    y_pred_mlp = scaler_y.inverse_transform(y_pred_mlp_scaled)

    # ------------------- 对比结果 -------------------
    print("\n=== 探针评估结果对比 ===")
    print(f"{'Label':<15} | {'Linear R²':<12} | {'MLP R²':<12} | {'结论'}")
    print("-" * 65)

    for i, label in enumerate(labels):
        r2_lin = r2_score(y_test[:, i], y_pred_linear[:, i])
        r2_mlp = r2_score(y_test[:, i], y_pred_mlp[:, i])

        # 判断逻辑
        if r2_mlp > 0.9:
            conclusion = "✅ 信息存在 (非线性编码)"
        elif r2_mlp > r2_lin + 0.1:
            conclusion = "⚠️ 强非线性信息"
        elif r2_mlp > 0.7:
            conclusion = "🟡 部分信息存在"
        elif r2_mlp < 0:
            conclusion = "❌ MLP 过拟合失败"
        else:
            conclusion = "➖ 线性为主"

        print(f"{label:<15} | {r2_lin:<12.4f} | {r2_mlp:<12.4f} | {conclusion}")


if __name__ == "__main__":
    pretrained_path = Path(
        "outputs/train/diffusion_pusht_augmented/checkpoints/last/pretrained_model"
    )
    if not pretrained_path.exists():
        print(f"错误：找不到预训练模型路径 {pretrained_path}")
    else:
        # 第一步：收集数据 (建议跑 10-20 个 episode 获取足够数据)
        X, y = collect_data_for_probe(pretrained_path, n_episodes=30)
        # 第二步：训练与评估
        if len(X) > 0:
            train_and_evaluate_probe(X, y)
