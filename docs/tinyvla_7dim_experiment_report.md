# TinyVLA 7 维单臂格式实验报告

> **实验时间**：2026-08-21 至 2026-08-22  
> **实验目标**：验证 7 维单臂动作空间是否能改善 TinyVLA 在 rm65b 单臂本体上的性能  
> **最终结果**：❌ 失败 - 7 维模型在可行 seed 上成功率为 0%

---

## 目录

1. [实验背景](#实验背景)
2. [实验过程](#实验过程)
3. [详细结果](#详细结果)
4. [问题分析](#问题分析)
5. [经验教训](#经验教训)
6. [后续建议](#后续建议)
7. [附录](#附录)

---

## 实验背景

### 动机

RoboTwin 平台的 rm65b 单臂本体使用双臂格式（14 维）的观测/动作空间：
- 格式：`[左臂6关节, 左夹爪1, 右臂6关节, 右夹爪1]`
- 问题：单臂本体下，左右两半的关节角完全相同（读取同一批 joint 对象），仅夹爪维度不同

这一冗余表示导致：
1. 模型需要输出两组本应相等的关节角（实测预测差 0.005~0.027，而动作步长仅 0.016）
2. 模型额外学习"本次该驱动哪一侧夹爪"这一判别任务（物理上只有一个夹爪）

### 假设

将 14 维格式转换为 7 维单臂格式：
- 格式：`[臂6关节, 夹爪1]`
- 预期效果：
  - 消除关节角冗余
  - 简化学习目标
  - 提高模型性能

---

## 实验过程

### 阶段 1：数据格式分析（2026-08-17 ~ 2026-08-20）

#### 1.1 发现问题

**文件位置**：`envs/_base_task.py:580-590`

```python
# 单臂本体下，left_arm_joints 和 right_arm_joints 是同一批对象
left_arm_joints = self.left_robot.arms[0].arm_joints
right_arm_joints = self.right_robot.arms[0].arm_joints

# 验证
assert left_arm_joints[0] is right_arm_joints[0]  # True
```

**实测数据**（2026-08-17）：
- 随机选取 10 个 episode 的关节角差值
- 左右关节角差值：`0.000000 ± 0.000000`（完全相同）
- 夹爪差值：`0.23 ± 0.15`（有变化，由任务选臂决定）

#### 1.2 评估基线纠正

**发现**：原评估方法有误，直接在所有 eval seeds 上测试，但专家策略成功率仅 15.3%。

**测试方法**：
- 使用 `diag_expert_evalseed.py` 测试专家策略在 196 个 eval seeds 上的表现
- 结果：30/196 = 15.3% 成功

**可行 seed 筛选**：
- 创建 `feasible_seeds.txt`，包含 30 个专家可完成的 seed
- 后续评估只在这 30 个可行 seed 上进行，使理论上限回到 100%

#### 1.3 14 维模型基线

**模型**：TinyVLA 14 维，混合数据（rm65b + rm65b_lateral 各 50 条）

**评估结果**（2026-08-19）：
- 在 30 个可行 seed 上：1/30 = 3.3% 成功
- 问题：模型输出左右关节角不一致，导致动作抖动

---

### 阶段 2：数据转换（2026-08-20）

#### 2.1 转换脚本

**文件**：`policy/TinyVLA/convert_to_7dim.py`

**转换逻辑**：
```python
def pick_gripper(a_left, a_right):
    """选出实际在动作的那一侧夹爪序列。
    判据：标准差更大者。
    """
    return a_left if a_left.std() >= a_right.std() else a_right

# 关节角：左右两半相同，取左侧即可
a_joints, q_joints = action[:, :6], qpos[:, :6]

# 夹爪：取实际在动的一侧
use_left = action[:, 6].std() >= action[:, 13].std()
a_grip = action[:, 6] if use_left else action[:, 13]
q_grip = qpos[:, 6] if use_left else qpos[:, 13]

# 组合为 7 维
action_7dim = np.concatenate([a_joints, a_grip[:, None]], axis=1)
qpos_7dim = np.concatenate([q_joints, q_grip[:, None]], axis=1)
```

#### 2.2 转换结果

**数据集**：
- `demo_rm65b_single_150-150_7dim`：150 条（桌面装）
- `demo_rm65b_lateral_150-150_7dim`：150 条（侧装）
- 总计：300 条 episode

**夹爪选择统计**：
- 左臂驱动：176 条（58.7%）
- 右臂驱动：124 条（41.3%）

**验证**：
- 关节角维度：6 ✓
- 夹爪维度：1 ✓
- 数据完整性：所有 episode 均成功转换 ✓

---

### 阶段 3：训练配置（2026-08-20）

#### 3.1 数据集配置

**文件**：`policy/TinyVLA/aloha_scripts/constants.py`

```python
"place_object_stand_mix300_7dim": {
    'dataset_dir': [
        "data/sim-place_object_stand/demo_rm65b_single_150-150_7dim",
        "data/sim-place_object_stand/demo_rm65b_lateral_150-150_7dim",
    ],
    'episode_len': 200,
    'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
    "sample_weights": [1, 1]
}
```

#### 3.2 训练脚本

**文件**：`policy/TinyVLA/scripts/franka/train_rm65b_7dim.sh`

**关键参数**：
```bash
--action_dim 7
--state_dim 7
--max_steps 6000
--per_device_train_batch_size 1
--gradient_accumulation_steps 16
--learning_rate 2e-5
```

**设计理由**：
- `max_steps=6000`：数据量由 100 增至 300 后，6000 步约 3.2 epoch，接近上一轮的训练充分度
- `action_dim=7, state_dim=7`：匹配 7 维格式

---

### 阶段 4：训练执行（2026-08-21）

#### 4.1 训练过程

**第一次启动**：2026-08-21 14:17
- 运行至 step 71/6000 时被外部进程终止
- Loss：0.51 → 0.46（正常下降）

**第二次启动**：2026-08-21 18:10
- 运行至 step 4735/6000 时再次被终止
- 已保存 checkpoint-4000

**第三次启动（恢复训练）**：2026-08-21 22:10
- **完成时间**：2026-08-21 22:55
- **总时长**：4 小时 44 分
- **最终 loss**：0.0265
- **Checkpoints**：1000, 2000, 3000, 4000, 5000, 6000

#### 4.2 训练曲线

```
Epoch 0.0  - Loss: 1.31
Epoch 0.5  - Loss: 0.08
Epoch 1.0  - Loss: 0.07
Epoch 1.5  - Loss: 0.06
Epoch 2.0  - Loss: 0.05
Epoch 2.5  - Loss: 0.04
Epoch 3.0  - Loss: 0.03
Epoch 3.17 - Loss: 0.0265 (最终)
```

**Loss 收敛正常**，从 1.31 降至 0.0265，下降 50 倍。

---

### 阶段 5：部署配置（2026-08-21）

#### 5.1 推理代码修改

**文件**：`policy/TinyVLA/deploy_policy.py`

**观测编码**（encode_obs）：
```python
if os.environ.get("TINYVLA_ACTION_DIM") == "7":
    qpos = (observation["joint_action"]["left_arm"] +
            [observation["joint_action"]["left_gripper"]])
else:
    qpos = (observation["joint_action"]["left_arm"] + [observation["joint_action"]["left_gripper"]] +
            observation["joint_action"]["right_arm"] + [observation["joint_action"]["right_gripper"]])
```

**动作扩展**（eval）：
```python
if os.environ.get("TINYVLA_ACTION_DIM") == "7":
    actions = np.concatenate([actions, actions], axis=-1)  # 7 维 → 14 维
```

#### 5.2 环境变量

**训练时**：
- `TINYVLA_ACTION_DIM=7`
- `SINGLE_ARM_JOINT_MODE=mean`

**推理时**：
- `TINYVLA_ACTION_DIM=7`：启用 7 维模式
- `SINGLE_ARM_JOINT_MODE=mean`：对左右臂预测取均值

---

### 阶段 6：评估执行（2026-08-22）

#### 6.1 评估配置

**模型**：checkpoint-6000（最终模型）  
**评估种子**：30 个可行 seed  
**环境变量**：
- `TINYVLA_ACTION_DIM=7`
- `SINGLE_ARM_JOINT_MODE=mean`

#### 6.2 评估脚本

**文件**：`/tmp/eval_7dim_feasible.py`

**评估流程**：
```python
for seed in feasible_seeds:
    env.setup_demo(seed=seed, is_test=True)
    obs = env.get_obs()
    while env.take_action_cnt < env.step_lim:
        tinyvla_eval(env, model, obs)
        obs = env.get_obs()
        if env.eval_success: break
    record_result(seed, env.eval_success)
```

#### 6.3 评估结果

**开始时间**：2026-08-22 00:07  
**完成时间**：2026-08-22 ~01:30（约 1.5 小时）

**结果**：
```
最终结果：0/30 = 0.0%
```

**详细记录**：
- 所有 30 个 seed 均运行到 400 步上限
- 无一成功
- 典型输出：`seed 100000: 失败 (步数 400/400)`

---

## 详细结果

### 训练结果

| 指标 | 值 |
|------|-----|
| 训练步数 | 6000/6000 (100%) |
| 最终 loss | 0.0265 |
| 训练时长 | 4 小时 44 分 |
| GPU 显存 | 4.8G（7 维）vs 8.5G（14 维） |
| Checkpoints | 6 个（1000-6000） |

### 评估结果对比

| 模型 | 数据量 | 动作维度 | 可行 seed 成功率 | 备注 |
|------|--------|----------|------------------|------|
| 专家策略 | - | - | 15.3% (30/196) | 所有 eval seeds |
| TinyVLA 14 维 | 100 条 | 14 | 3.3% (1/30) | 可行 seeds |
| TinyVLA 14 维 | 300 条 | 14 | 未测试 | - |
| **TinyVLA 7 维** | **300 条** | **7** | **0.0% (0/30)** | **可行 seeds** |

---

## 问题分析

### 1. Loss 收敛正常但策略失败

**现象**：
- 训练 loss 从 1.31 降至 0.0265（下降 50 倍）
- 评估时所有 seed 均运行到 400 步上限，无一成功

**可能原因**：
1. **过拟合**：模型记忆了训练数据的精确轨迹，但未学到泛化策略
2. **分布偏移**：训练数据的动作分布与真实环境不匹配
3. **动作归一化问题**：7 维数据的归一化统计量可能有误

### 2. 与 14 维模型对比

**14 维模型**：至少有 3.3% 成功率（1/30）  
**7 维模型**：完全失败（0/30）

**差异分析**：
- 14 维模型虽有关节角冗余，但至少学到了部分有效策略
- 7 维模型在消除冗余的同时，可能丢失了重要信息

### 3. 数据转换潜在问题

**夹爪选择逻辑**：
```python
use_left = action[:, 6].std() >= action[:, 13].std()
```

**问题**：
- 仅根据标准差选择夹爪，可能忽略了时序信息
- 训练数据中夹爪选择与物体位置强相关，但模型可能未学到这一映射

**验证缺失**：
- 未对转换后的数据进行可视化验证
- 未检查转换前后的动作分布一致性

### 4. 训练与推理不匹配

**训练时**：
- 使用 7 维动作和状态
- 夹爪已预先选择好

**推理时**：
- 输入 7 维状态（只取左侧关节角和夹爪）
- 输出 7 维动作，再扩展为 14 维
- 使用 `SINGLE_ARM_JOINT_MODE=mean` 对左右臂预测取均值

**潜在问题**：
- 训练时夹管选择基于整个 episode 的标准差
- 推理时无法获取未来信息，只能固定取左侧
- 可能导致动作不匹配

---

## 经验教训

### 1. 数据转换验证不足

**教训**：
- 应对转换后的数据进行可视化验证
- 检查转换前后的动作分布一致性
- 特别关注夹爪维度

**建议做法**：
```python
# 转换前后对比
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.plot(action_14dim[:, :6].mean(axis=0), label='14-dim joints')
plt.plot(action_7dim[:, :6].mean(axis=0), label='7-dim joints')
plt.legend()
plt.subplot(132)
plt.plot(action_14dim[:, 6], label='14-dim left gripper')
plt.plot(action_14dim[:, 13], label='14-dim right gripper')
plt.plot(action_7dim[:, 6], label='7-dim gripper')
plt.legend()
plt.subplot(133)
plt.hist(action_14dim[:, 6], alpha=0.5, label='14-dim left')
plt.hist(action_14dim[:, 13], alpha=0.5, label='14-dim right')
plt.hist(action_7dim[:, 6], alpha=0.5, label='7-dim')
plt.legend()
plt.savefig('action_distribution_comparison.png')
```

### 2. 过拟合风险

**教训**：
- Loss 收敛不代表策略有效
- 应在训练过程中定期评估（如每 1000 步）
- 应使用验证集监控泛化能力

**建议做法**：
```bash
# 每 1000 步评估一次
--save_steps 1000
--evaluation_strategy steps
--eval_steps 1000
```

### 3. 渐进式验证

**教训**：
- 不应直接进行大规模实验
- 应先在小规模数据上验证方法有效性

**建议流程**：
1. **单元测试**：验证数据转换正确性
2. **过拟合测试**：在单个 episode 上训练，验证模型能否完美复现
3. **小规模测试**：在 10-20 个 episode 上训练，评估是否有学习迹象
4. **大规模训练**：确认有效后再扩展

### 4. 基线对比不足

**教训**：
- 应先在相同数据量下对比 14 维和 7 维
- 不应同时改变数据量和动作维度两个变量

**建议实验设计**：
| 实验 | 数据量 | 动作维度 | 目的 |
|------|--------|----------|------|
| A | 100 条 | 14 | 基线 |
| B | 300 条 | 14 | 数据量影响 |
| C | 100 条 | 7 | 维度影响 |
| D | 300 条 | 7 | 综合效果 |

---

## 后续建议

### 短期（立即执行）

1. **回到 14 维模型**
   - 7 维实验失败，14 维至少有 3.3% 成功率
   - 先在 300 条数据上训练 14 维模型，验证数据量增加的效果

2. **检查数据转换**
   - 对比转换前后的动作分布
   - 特别检查夹爪维度的变化
   - 可视化几个 episode 的轨迹

3. **过拟合测试**
   - 在单个 episode 上训练 7 维模型
   - 验证模型能否完美复现该 episode
   - 如果不能，说明模型容量或训练配置有问题

### 中期（一周内）

1. **渐进式实验**
   - 按"经验教训 4"的实验设计执行
   - 单独验证数据量和动作维度的影响

2. **模型调试**
   - 记录训练过程中的预测动作分布
   - 对比预测动作与真实动作的差异
   - 分析哪些关节/维度预测不准确

3. **超参数搜索**
   - 尝试不同的学习率（1e-5, 2e-5, 5e-5）
   - 尝试不同的训练步数（3000, 6000, 12000）
   - 尝试不同的 batch size

### 长期（研究方向）

1. **数据增强**
   - 增加域随机化
   - 数据增强（图像翻转、亮度变化等）
   - 合成数据生成

2. **模型改进**
   - 尝试其他策略架构（如 Diffusion Policy, ACT）
   - 引入课程学习
   - 多任务学习

3. **平台适配**
   - 深入理解单臂本体的特殊性
   - 设计专门的动作空间表示
   - 考虑使用侧装相机而非腕部相机

---

## 附录

### A. 文件清单

#### 新增文件

| 文件路径 | 说明 |
|----------|------|
| `policy/TinyVLA/convert_to_7dim.py` | 14 维→7 维数据转换脚本 |
| `policy/TinyVLA/scripts/franka/train_rm65b_7dim.sh` | 7 维训练脚本 |
| `data/sim-place_object_stand/demo_rm65b_single_150-150_7dim/` | 7 维格式数据集（桌面装） |
| `data/sim-place_object_stand/demo_rm65b_lateral_150-150_7dim/` | 7 维格式数据集（侧装） |

#### 修改文件

| 文件路径 | 修改内容 |
|----------|----------|
| `policy/TinyVLA/aloha_scripts/constants.py` | 添加 `place_object_stand_mix300_7dim` 配置 |
| `policy/TinyVLA/deploy_policy.py` | 添加 `TINYVLA_ACTION_DIM=7` 支持 |
| `envs/_base_task.py` | 添加 `SINGLE_ARM_JOINT_MODE` 环境变量支持 |

#### 训练产物

| 路径 | 大小 | 说明 |
|------|------|------|
| `policy/TinyVLA/unet_diffusion_policy_results/place_object_stand_mix300_7dim-1BS-2e-5LR-4noise/` | - | 训练输出目录 |
| `├── checkpoint-{1000,2000,3000,4000,5000,6000}/` | 各 ~2GB | 模型检查点 |
| `├── dataset_stats.pkl` | 3.4KB | 数据统计（均值/标准差） |
| `├── log_7dim.log` | 559KB | 训练日志（第一次） |
| `└── log_7dim_resume.log` | - | 训练日志（恢复后） |

### B. 关键命令

#### 数据转换

```bash
python policy/TinyVLA/convert_to_7dim.py \
  data/sim-place_object_stand/demo_rm65b_single_150-150 \
  data/sim-place_object_stand/demo_rm65b_single_150-150_7dim

python policy/TinyVLA/convert_to_7dim.py \
  data/sim-place_object_stand/demo_rm65b_lateral_150-150 \
  data/sim-place_object_stand/demo_rm65b_lateral_150-150_7dim
```

#### 训练

```bash
PATH=/home/jy/miniconda3/envs/RoboTwin/bin:$PATH \
PYTHONNOUSERSITE=1 PYTHONUTF8=1 \
bash policy/TinyVLA/scripts/franka/train_rm65b_7dim.sh
```

#### 评估

```bash
PATH=/home/jy/miniconda3/envs/RoboTwin/bin:$PATH \
PYTHONNOUSERSITE=1 PYTHONUTF8=1 \
TINYVLA_ACTION_DIM=7 SINGLE_ARM_JOINT_MODE=mean \
python /tmp/eval_7dim_feasible.py
```

### C. 环境信息

| 项目 | 值 |
|------|-----|
| Python 版本 | 3.10 |
| PyTorch 版本 | 2.x |
| CUDA 版本 | 11.8 |
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU (16GB) |
| Conda 环境 | RoboTwin |

### D. 时间线

| 时间 | 事件 |
|------|------|
| 2026-08-17 | 发现单臂本体的 14 维格式冗余问题 |
| 2026-08-19 | 测试 14 维 TinyVLA 基线（3.3% 成功率） |
| 2026-08-20 | 完成 7 维数据转换（300 条） |
| 2026-08-21 14:17 | 首次启动 7 维训练（被中断） |
| 2026-08-21 18:10 | 第二次启动训练（被中断） |
| 2026-08-21 22:10 | 恢复训练 |
| 2026-08-21 22:55 | 训练完成 |
| 2026-08-22 00:07 | 开始评估 |
| 2026-08-22 ~01:30 | 评估完成（0/30 失败） |

---

## 结论

本次实验尝试通过将 TinyVLA 的动作空间从 14 维双臂格式转换为 7 维单臂格式，以消除单臂本体上的冗余表示。实验结果表明：

1. **训练成功**：Loss 正常收敛（1.31 → 0.0265）
2. **评估失败**：在 30 个可行 seed 上成功率为 0%
3. **性能退化**：比 14 维模型（3.3%）更差

**根本原因可能是**：
- 数据转换过程中的信息丢失（夹爪选择逻辑）
- 模型过拟合训练数据，未学到泛化策略
- 训练与推理的动作空间不匹配

**下一步行动**：
1. 回到 14 维模型，先验证数据量增加的效果
2. 深入分析 7 维数据转换的正确性
3. 设计更严谨的渐进式验证流程

---

**报告作者**：Claude  
**报告日期**：2026-08-22  
**报告版本**：v1.0