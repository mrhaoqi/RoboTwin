"""RoboTwin 评估适配层：SmolVLA（LeRobot 训练产物）。

用一份代码同时支持两种运行模式：

  单进程   script/eval_policy.py          —— 需要同一环境里既有 sapien 又有 lerobot
  C/S 分离 script/policy_model_server.py  —— 仿真在 RoboTwin 环境, 推理在 lerobot 环境
                                             ⚠️ 本项目必须用这种, 见下

## 为什么必须走 C/S

两套依赖装不进同一个环境：

    RoboTwin 环境  sapien 3.0.0b1 / mplib / curobo    transformers 4.45.2
    lerobot 环境   lerobot 0.4.4                      transformers 4.57.6

lerobot 0.4.4 要求 transformers>=4.57.1，而把 RoboTwin 环境的 transformers 从
4.45.2 升上去会波及 TinyVLA / DexVLA（它们依赖 InternVL 的特定行为，实验报告里
记过 auto_map 冲突）。两边 torch 同为 2.10.0+cu128，但这解决不了 transformers。

## 三个关键约定（错一个就静默失效）

1. **7 维 ↔ 14 维**。RoboTwin 的 take_action(action_type='qpos') 收的是
   `[左臂6 + 左夹爪 + 右臂6 + 右夹爪]`。单臂本体下左右指向同一批物理关节，
   故模型的 7 维输出复制展开即可，与训练数据的构造方式互逆。

2. **图像通道序**。仿真给的是 RGB 原序，转换器（修正后）也不翻通道，
   这里同样不翻。翻错了不报错，只让预训练视觉先验失效。

3. **必须过 postprocessor**。policy.select_action 返回的是**归一化空间**的动作，
   反归一化在 postprocessor 里。只取 preprocessor 会让输出与真实关节角差一个
   尺度 —— 症状是各维动作标准差整齐地接近 1.0。

## 动作块的消费方式

SmolVLA 内部维护一个长度 n_action_steps(默认 50) 的队列：select_action 每次弹一个，
队列空了才重新推理。所以每个仿真步调一次 get_action 是对的 —— 推理被摊薄到
每 50 步一次（实测单次约 1.4 s，弹出约 10 ms）。
reset() 清空队列，必须在每条 episode 开始时调用。
"""

import os

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _call(model, fn, obs=None):
    """兼容单进程对象与 ModelClient 代理两种形态。"""
    if model.__class__.__name__ == "ModelClient":
        return model.call(fn, obs)
    method = getattr(model, fn)
    return method(obs) if obs is not None else method()


def encode_obs(observation):
    """RoboTwin 观测 → 适配层需要的最小集合。

    只取头部 + 腕部两路。原始观测里有 head/left/right 三路，但单臂本体下
    left_camera 与 right_camera 逐字节相同（同一个腕部相机），第三路是
    双臂模板的产物，不含新信息。
    """
    obs = observation["observation"]
    return {
        "head": np.asarray(obs["head_camera"]["rgb"], dtype=np.uint8),
        "wrist": np.asarray(obs["left_camera"]["rgb"], dtype=np.uint8),
        # joint_action/vector 与训练数据同源（都来自 get_*_arm_jointState）
        "state14": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
    }


class SmolVLAAdapter:
    def __init__(self, ckpt_path, device="cuda", n_action_steps=None):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        ckpt = os.path.expanduser(ckpt_path)
        self.policy = SmolVLAPolicy.from_pretrained(ckpt)
        self.policy.to(device)
        self.policy.eval()
        # pre 与 post 都要 —— post 负责反归一化, 见模块说明第 3 条
        self.pre, self.post = make_pre_post_processors(self.policy.config,
                                                       pretrained_path=ckpt)
        self.device = device

        cfg = self.policy.config
        if n_action_steps:
            # 缩短动作块 = 更频繁重规划(闭环更紧), 代价是推理更频繁
            cfg.n_action_steps = int(n_action_steps)
        self.img_keys = sorted(k for k in cfg.input_features
                               if k.startswith("observation.images."))
        self.state_dim = cfg.input_features["observation.state"].shape[0]
        # 期望的图像尺寸取自训练时的 features, 与仿真实际输出可能不同, 需缩放
        self.img_hw = {k: tuple(cfg.input_features[k].shape[1:3]) for k in self.img_keys}
        print(f"[smolvla] 已加载 {ckpt}")
        print(f"[smolvla] state {self.state_dim} 维, 图像槽 {self.img_keys}, "
              f"chunk {cfg.n_action_steps}")

    # ---- 供 server 通过 cmd 调用的方法 ----

    def reset(self):
        """每条 episode 开始时清空动作队列。不清会把上一条的残留动作发出去。"""
        self.policy.reset()
        return True

    def get_action(self, obs):
        import cv2
        torch = self.torch

        # 7 维 state: 关节取左半(左右为同一批物理关节), 夹爪取下标 6。
        # 评估时取哪侧都行 —— 我们下发的 14 维动作把两侧夹爪设成同一个值,
        # 首帧之前两侧也都停在初始值。这与训练时"取实际在动的那侧"不冲突:
        # 那是因为专家只驱动了一侧, 而这里两侧总是同步的。
        s14 = np.asarray(obs["state14"], dtype=np.float32).reshape(-1)
        state = np.concatenate([s14[:6], s14[6:7]]).astype(np.float32)

        raw = {"observation.state": torch.from_numpy(state),
               "task": obs.get("task") or ""}
        for k, src in zip(self.img_keys, [obs["head"], obs["wrist"]]):
            img = np.asarray(src, dtype=np.uint8)
            h, w = self.img_hw[k]
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            # HWC uint8 RGB → CHW float[0,1]。**不翻通道**, 见模块说明第 2 条
            raw[k] = torch.from_numpy(
                np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32) / 255.0)

        with torch.no_grad():
            act = self.policy.select_action(self.pre(raw))
        act = self.post(act)
        if not isinstance(act, torch.Tensor):
            act = torch.as_tensor(act)
        return act.detach().float().cpu().numpy().reshape(-1).tolist()


def get_model(usr_args):
    return SmolVLAAdapter(
        ckpt_path=usr_args["ckpt_path"],
        device=usr_args.get("device", "cuda"),
        n_action_steps=usr_args.get("n_action_steps"),
    )


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    obs["task"] = TASK_ENV.get_instruction()

    a7 = np.asarray(_call(model, "get_action", obs), dtype=np.float32).reshape(-1)
    if a7.shape[0] != 7:
        raise ValueError(f"期望 7 维动作, 实得 {a7.shape}")

    # 7 → 14: 单臂本体下左右是同一批物理关节, 两侧下发相同值。
    # 这与 take_action 的切片 [:6] / [6] / [7:13] / [13] 对应。
    a14 = np.concatenate([a7[:6], a7[6:7], a7[:6], a7[6:7]]).astype(np.float32)
    TASK_ENV.take_action(a14, action_type="qpos")


def reset_model(model):
    _call(model, "reset")
