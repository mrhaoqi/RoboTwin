# TinyVLA 7 维单臂实验工作日志

> 实验周期：2026-08-17 至 2026-08-22  
> 最后更新：2026-08-22

---

## 2026-08-17（周六）

### 问题发现

**时间**：下午

**事件**：分析 rm65b 单臂本体的观测/动作空间

**发现**：
- 单臂本体使用 14 维双臂格式：`[左臂6, 左夹爪, 右臂6, 右夹爪]`
- 左右两半的关节角完全相同（读取同一批 joint 对象）
- 仅夹爪维度不同（由任务选臂决定）

**验证**：
```python
# envs/_base_task.py
left_arm_joints = self.left_robot.arms[0].arm_joints
right_arm_joints = self.right_robot.arms[0].arm_joints
assert left_arm_joints[0] is right_arm_joints[0]  # True
```

**实测数据**：
- 随机选取 10 个 episode
- 关节角差值：`0.000000 ± 0.000000`
- 夹爪差值：`0.23 ± 0.15`

**结论**：14 维格式存在严重冗余

---

## 2026-08-19（周一）

### 评估基线纠正

**时间**：上午

**事件**：发现原评估方法有误

**问题**：
- 原方法在所有 eval seeds 上测试（seed=100000 起顺序）
- 专家策略成功率仅 15.3%（30/196）
- 模型成功率的理论上限约为 15% 而非 100%

**解决方案**：
- 创建 `diag_expert_evalseed.py` 测试专家策略
- 筛选 30 个可行 seed（`feasible_seeds.txt`）
- 后续评估只在这 30 个 seed 上进行

**结果**：
- 专家成功率：15.3%（30/196，所有 eval seeds）
- 14 维 TinyVLA：3.3%（1/30，可行 seeds）

---

## 2026-08-20（周二）

### 数据转换脚本开发

**时间**：全天

**工作内容**：

1. **编写转换脚本** (`convert_to_7dim.py`)
   - 关节角：取左侧（左右相同）
   - 夹爪：取标准差更大的一侧
   - 保留：action, qpos, language_raw, images

2. **执行转换**
   ```bash
   python convert_to_7dim.py \
     data/sim-place_object_stand/demo_rm65b_single_150-150 \
     data/sim-place_object_stand/demo_rm65b_single_150-150_7dim
   
   python convert_to_7dim.py \
     data/sim-place_object_stand/demo_rm65b_lateral_150-150 \
     data/sim-place_object_stand/demo_rm65b_lateral_150-150_7dim
   ```

3. **验证结果**
   - 转换成功：300 条 episode
   - 数据完整性：所有 episode 均有效
   - 夹爪选择：左 176 条（58.7%），右 124 条（41.3%）

### 训练配置

**时间**：下午

**工作内容**：

1. **数据集配置** (`aloha_scripts/constants.py`)
   - 添加 `place_object_stand_mix300_7dim` 配置
   - 数据集：rm65b_single + rm65b_lateral 各 150 条

2. **训练脚本** (`scripts/franka/train_rm65b_7dim.sh`)
   - `--action_dim 7 --state_dim 7`
   - `--max_steps 6000`（约 3.2 epoch）
   - 输出目录：`unet_diffusion_policy_results/place_object_stand_mix300_7dim-*`

---

## 2026-08-21（周三）

### 训练执行

#### 第一次启动

**时间**：14:17  
**结果**：71/6000 步后被外部进程终止  
**Log**：`log_7dim.log`  
**Loss**：0.51 → 0.46

#### 第二次启动

**时间**：18:10  
**结果**：4735/6000 步后被终止  
**已保存**：checkpoint-4000

#### 第三次启动（恢复训练）

**时间**：22:10  
**完成**：22:55  
**总时长**：4 小时 44 分  
**最终 loss**：0.0265  
**Checkpoints**：1000, 2000, 3000, 4000, 5000, 6000

### 部署配置

**时间**：晚上

**工作内容**：

1. **修改 `deploy_policy.py`**
   - 添加 `TINYVLA_ACTION_DIM=7` 支持
   - 观测编码：只取左侧关节角和夹爪
   - 动作扩展：7 维 → 14 维

2. **修改 `envs/_base_task.py`**
   - 添加 `SINGLE_ARM_JOINT_MODE` 环境变量
   - 支持 `mean`, `left`, `right` 三种模式

---

## 2026-08-22（周四）

### 评估执行

**开始时间**：00:07  
**完成时间**：~01:30  
**耗时**：约 1.5 小时

**评估配置**：
- 模型：checkpoint-6000
- Seeds：30 个可行 seed
- 环境变量：`TINYVLA_ACTION_DIM=7`, `SINGLE_ARM_JOINT_MODE=mean`

**结果**：
```
最终结果：0/30 = 0.0%
```

**详细记录**：
- 所有 30 个 seed 均运行到 400 步上限
- 无一成功
- 典型输出：`seed 100000: 失败 (步数 400/400)`

### 问题分析

**时间**：上午

**发现**：
1. Loss 收敛正常（0.0265）但策略完全失败
2. 比 14 维模型（3.3%）更差
3. 可能原因：过拟合、数据转换问题、训练与推理不匹配

**记录**：
- 创建详细实验报告：`docs/tinyvla_7dim_experiment_report.md`
- 创建工作日志：`docs/tinyvla_7dim_work_log.md`

---

## 待办事项

### 短期（立即）

- [ ] 在 300 条数据上训练 14 维 TinyVLA，验证数据量增加的效果
- [ ] 对比转换前后的动作分布（可视化验证）
- [ ] 在单个 episode 上进行过拟合测试

### 中期（一周内）

- [ ] 按渐进式实验设计执行对比实验
- [ ] 分析预测动作与真实动作的差异
- [ ] 尝试不同超参数

### 长期

- [ ] 数据增强策略
- [ ] 模型架构改进
- [ ] 平台适配优化

---

## 文件变更清单

### 新增文件

- `policy/TinyVLA/convert_to_7dim.py`
- `policy/TinyVLA/scripts/franka/train_rm65b_7dim.sh`
- `data/sim-place_object_stand/demo_rm65b_single_150-150_7dim/`（150 条）
- `data/sim-place_object_stand/demo_rm65b_lateral_150-150_7dim/`（150 条）
- `docs/tinyvla_7dim_experiment_report.md`
- `docs/tinyvla_7dim_work_log.md`（本文件）

### 修改文件

- `policy/TinyVLA/aloha_scripts/constants.py`
- `policy/TinyVLA/deploy_policy.py`
- `envs/_base_task.py`

### 训练产物

- `policy/TinyVLA/unet_diffusion_policy_results/place_object_stand_mix300_7dim-1BS-2e-5LR-4noise/`
  - checkpoint-{1000,2000,3000,4000,5000,6000}/
  - dataset_stats.pkl
  - log_7dim.log, log_7dim_resume.log

---

**日志结束**：2026-08-22  
**状态**：暂停，等待进一步调查