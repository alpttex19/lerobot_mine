# 修改记录（Change Log）

本文档记录对 lerobot 原始代码库的所有定制化修改，按提交时间倒序排列。



---

## [c56e250] 添加目标图像辅助监督 · 2026-04-17

**提交说明：** `add goal image`

**核心目标：** 在 Diffusion Policy 的训练中引入目标图像辅助损失，使 RGB 编码器被迫学习包含目标位姿信息的特征表示。

### `src/lerobot/policies/diffusion/modeling_diffusion.py`
- **新增 `goal_encoder`（`DiffusionRgbEncoder`）和 `goal_pred_head`（`nn.Linear`）**：构成目标预测辅助分支，仅在训练时激活。
- **新增辅助损失计算逻辑**：在 `compute_loss` 中，若 batch 包含 `goal.<cam>` 键，则提取目标图像特征并用当前帧特征预测之，计算 MSE 损失，以 `0.1` 的权重叠加到主损失上。
- **新增目标图像归一化**：在 `forward` 中对 `goal.<cam>` 图像做 batch 级 z-score 归一化，并对齐到观测图像的均值/标准差，确保两路编码器输入尺度一致。
- **`forward` 返回值**：现在返回 `(loss, output_dict)`，其中 `output_dict` 包含 `goal_aux_loss` 用于 wandb 日志记录。

### `src/lerobot/datasets/dataset_reader.py`
- **新增目标帧读取逻辑**：在 `__getitem__` 中，取当前 episode 最后一帧作为目标图像，以 `goal.<cam>` 键存入样本。推理时 batch 中无此键，辅助分支自动跳过，对推理无侵入。

### `src/lerobot/datasets/pusht_augment.py`
- **新增目标图像增强**：在 `AugmentCollate` 中，对 batch 内 `goal.*` 键的图像应用与主图像完全相同的随机仿射变换（同 batch index 同变换参数），保证目标图像与观测图像的增强一致性。

### `src/lerobot/envs/wrappers.py`
- 小幅修正（见下一条）。

### `examples/training/eval_policy.py` / `test_feature.py`
- 更新 checkpoint 路径，`main` 函数签名重构为接受 `pretrained_path` 和 `video_dir` 参数。

---

## [a15d3c9] 添加线性探针脚本 · 2026-04-17

**提交说明：** `add linear probe`

**核心目标：** 验证训练后的 Diffusion Policy 的 RGB 编码器中间特征是否编码了目标位姿信息。

### 新增文件

**`examples/training/test_feature.py`**
- `FeatureExtractor`：使用 PyTorch forward hook 在指定层（`diffusion.rgb_encoder.backbone.7.1.conv2`）提取中间特征，并做全局平均池化降维。
- `collect_data_for_probe`：运行随机目标环境，收集 `(特征向量, 真实位姿)` 数据对；角度以 `sin/cos` 形式编码（避免角度周期性问题），位姿标签为 `[Block_X, Block_Y, Block_Sin, Block_Cos, Goal_X, Goal_Y, Goal_Sin, Goal_Cos]`（8 维）。
- `train_and_evaluate_probe`：分别训练**线性探针**（`LinearRegression`）和**非线性探针**（`MLPRegressor` + `StandardScaler` + L2 正则化），对比 R² 以判断信息是否以线性/非线性形式编码在特征中。

**`examples/training/visualize_feature.py`**
- 特征可视化辅助脚本。

---

## [f77277bb] 几何增强重构（正确版） · 2026-04-16

**提交说明：** `right augmented version`

**核心目标：** 修复初版增强中旋转方向不一致的 bug，将图像变换与坐标变换统一到同一个 `RigidTransform` 类中。

### `src/lerobot/datasets/pusht_augment.py`（重构）
- **新增 `RigidTransform` 类**：封装 2D 刚体变换（绕场景中心旋转 + 平移），同时提供 `apply_coords()`（坐标变换）和 `apply_image()`（图像变换）两个接口，确保两者使用完全相同的数学约定：
  ```
  正角度 = 顺时针（CW），与 TF.affine 行为一致
  x' = (x-cx)*cosθ - (y-cy)*sinθ + cx + tx
  y' = (x-cx)*sinθ + (y-cy)*cosθ + cy + ty
  ```
- 增强参数验证：对增强后坐标越界的样本跳过，保证坐标合法性。
- 添加 `matplotlib.use("Agg")` 避免无显示器环境报错。

### `src/lerobot/envs/wrappers.py`
- 添加 `task_description` 和 `task` 属性到 `RandomGoalWrapper`。

### 其他
- 删除根目录下的 `pusht_augment.py`（旧版原型脚本，已移入 `src/`）。
- 更新 `run.sh`，添加训练命令模板。

---

## [12f0b234] 增强配置解析器 · 2026-04-16

**提交说明：** `add cfg parser for augmentation`

**核心目标：** 将增强开关接入 draccus CLI 配置系统，可通过命令行参数控制。

### `src/lerobot/configs/train.py`
- 新增两个训练配置字段：
  - `use_augmentation: bool = True` — 控制是否启用 `AugmentCollate`
  - `use_random_goal: bool = False` — 控制是否对 PushT 环境使用 `RandomGoalWrapper`

### `src/lerobot/envs/configs.py`
- 为 `PushtEnv` 添加 `random_goal` 布尔配置项。

### `src/lerobot/envs/wrappers.py`
- 新增 `RandomGoalWrapper`：在每次 `reset()` 时随机采样目标位姿 `(x, y, θ)`，使训练/评估分布覆盖更广的目标位置。

### `src/lerobot/scripts/lerobot_train.py`
- 根据 `cfg.use_augmentation` 决定是否将 `AugmentCollate` 作为 DataLoader 的 `collate_fn`。
- 在 eval tracker 中新增 `avg_max_reward` 指标并同步至 wandb。

---

## [bd34d348] 初始增强实现 · 2026-04-16

**提交说明：** `add augmentation`

**核心目标：** 对 PushT 数据集引入几何增强（旋转 + 平移），保证图像与状态/动作坐标的一致性变换。

### 新增文件

**`src/lerobot/datasets/pusht_augment.py`**（初版）
- 坐标系说明：图像 96×96（像素），坐标 512×512，缩放比 `SCALE = 96/512 = 0.1875`。
- 核心函数：
  - `_rotate_image`、`_translate_image`：图像级变换。
  - `_rotate_coords`、`_translate_coords`：坐标级变换（state/action）。
  - `augment_sample`：对单个样本应用随机旋转 + 平移。
- `AugmentCollate`：实现 batch 级增强的 collate_fn，对 `observation.images.*`、`action`、`observation.state` 同步应用相同随机变换。

**`examples/training/eval_policy.py`**
- 评估脚本：加载 checkpoint，在随机目标 PushT 环境中跑 20 个 episode，记录成功率、奖励，并保存 mp4 视频。

**`examples/training/train_policy.py`**（扩展）
- 示例训练脚本，集成增强 collate_fn。

**`LEARN.md`**
- 项目学习笔记，记录 lerobot 关键模块原理。

**`arapat_readme.md`**
- 常用命令速查表（训练、评估、数据可视化）。

### 修改文件

**`src/lerobot/envs/configs.py`**
- `PushtEnv`：添加 `random_goal` 字段，支持随机目标模式。

**`src/lerobot/scripts/lerobot_train.py`**
- 集成 `AugmentCollate` 和 `RandomGoalWrapper`。

---

## 总结：改动架构图

```
lerobot/
├── src/lerobot/
│   ├── configs/train.py          ← 新增 use_augmentation, use_random_goal 字段
│   ├── datasets/
│   │   ├── pusht_augment.py      ← 核心：几何增强 + AugmentCollate（含目标图像）
│   │   └── dataset_reader.py     ← 新增：读取 episode 末帧作为 goal 图像
│   ├── envs/
│   │   ├── configs.py            ← 新增：PushtEnv.random_goal 字段
│   │   └── wrappers.py           ← 新增：RandomGoalWrapper
│   ├── policies/diffusion/
│   │   └── modeling_diffusion.py ← 新增：goal_encoder 辅助分支 + 辅助损失
│   └── scripts/lerobot_train.py  ← 修改：接入增强、随机目标、max_reward 日志
├── examples/training/
│   ├── eval_policy.py            ← 新增：评估脚本（随机目标 + 视频保存）
│   ├── test_feature.py           ← 新增：线性/MLP 探针，验证特征编码
│   └── visualize_feature.py      ← 新增：特征可视化
└── LEARN.md / arapat_readme.md   ← 新增：学习笔记与命令速查
```
