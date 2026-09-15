#!/usr/bin/env python3
"""RoboTwin 原始 hdf5 → LeRobot 数据集（单臂 7 维 + 双路相机）。

直接从 RoboTwin 的**原始采集数据**转，不经过 ACT/TinyVLA 的中间格式 ——
中间格式已经做过一次重采样和上采样，再转一道只会叠加误差且难追溯。

## 为什么是 7 维

RoboTwin 的任务代码围绕双臂设计，单臂本体通过"同一实体兼任左右臂"复用任务逻辑，
所以原始数据是 14 维。但左右两半并非独立数据：

  关节 dim0-5 / dim7-12   同一批物理关节的同一次读数, **逐字节相同**
  夹爪 dim6 / dim13       两个独立的 Python 缓存变量, 只有任务选中的那侧被更新

所以降到 7 维对关节是**无损**的（丢弃完全重复的一半），对夹爪则必须**逐条 episode
判断取哪侧** —— 任务按物体生成位置选左或右，实测约 2:1。取错侧会得到一个恒定值，
数据看着完整但抓取信号全丢，且训练时不会报任何错。

## 为什么是 2 路相机

原始数据有 head/left/right 三路，但单臂本体下 left 与 right **逐字节相同**
（同一个腕部相机）。真机也只有头部 + 腕部两路，故输出 2 路并按真机命名。

## 帧移约定

沿用 RoboTwin 自己的约定（`policy/ACT/process_data.py`）：

    observation.state[t] = state[t]
    action[t]            = state[t+1]        t = 0 .. N-2

即"下一帧状态作为本帧动作"。N 帧原始数据产出 N-1 帧。这与真机拖拽示教采集时
只能用 `state[t+1]` 当 action 是同一个约定，两批数据在这一点上可比。

用法:
    ~/miniconda3/envs/lerobot/bin/python convert_robotwin_to_lerobot.py \\
        --src ~/workspace/RoboTwin/data/place_object_stand/demo_rm65b_single_150 \\
        --repo-id rm65b/place_object_stand_single_150 \\
        --out ~/lerobot_datasets

背景见 机器人仓库 doc/architecture/VLA训练数据采集方案.md。
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

JOINT_NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper"]


def load_episode(path: Path):
    """读一条原始 episode，返回 14 维状态序列与两路图像的 JPEG 字节。"""
    with h5py.File(path, "r") as h:
        la = h["joint_action/left_arm"][:]
        ra = h["joint_action/right_arm"][:]
        lg = h["joint_action/left_gripper"][:]
        rg = h["joint_action/right_gripper"][:]
        head = h["observation/head_camera/rgb"][:]
        # 单臂下 left_camera 与 right_camera 完全相同, 取其一即可
        wrist = h["observation/left_camera/rgb"][:]
    return la, ra, lg, rg, head, wrist


def pick_gripper_side(lg: np.ndarray, rg: np.ndarray):
    """选出实际在动作的那侧夹爪。

    判据是标准差 —— 未被选中的一侧停留在初始值 1.0（张开）恒定不变。
    若两侧都无变化（任务全程不抓取, 如 turn_switch）, 返回 left 并标记 ambiguous:
    此时两者都是常量、取哪侧等价, 但这一维本身不含信息, 调用方应当知情。
    """
    ls, rs = float(lg.std()), float(rg.std())
    if ls < 1e-9 and rs < 1e-9:
        return "left", True
    return ("left", False) if ls >= rs else ("right", False)


def decode_bgr2rgb(buf, resize=None) -> np.ndarray:
    """JPEG 字节 → HWC uint8 RGB。

    cv2.imdecode 出来是 BGR, 必须翻通道 —— 这类错误不会报错, 只表现为
    模型在真机上对颜色的判断系统性偏移, 极难察觉。
    """
    img = cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("JPEG 解码失败")
    if resize is not None:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img[:, :, ::-1])


def load_instruction(inst_dir: Path, idx: int, key: str) -> str:
    f = inst_dir / f"episode{idx}.json"
    if not f.exists():
        return ""
    d = json.loads(f.read_text())
    lst = d.get(key) or d.get("seen") or []
    return lst[0] if lst else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="RoboTwin 采集目录, 形如 data/<task>/<config>")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--out", type=Path, default=None, help="数据集根目录")
    ap.add_argument("--fps", type=int, default=15,
                    help="⚠️ RoboTwin 未显式记录帧率; 取自采集配置的 save_freq")
    ap.add_argument("--episodes", type=int, default=0, help="只转前 N 条, 0=全部")
    ap.add_argument("--instruction-key", default="seen", choices=["seen", "unseen"])
    ap.add_argument("--resize", default="", help="如 640x480; 留空保持原分辨率")
    ap.add_argument("--robot-type", default="rm65b")
    args = ap.parse_args()

    data_dir = args.src / "data"
    if not data_dir.is_dir():
        print(f"找不到 {data_dir}", file=sys.stderr)
        return 1
    inst_dir = args.src / "instructions"

    eps = sorted(data_dir.glob("episode*.hdf5"),
                 key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
    if args.episodes:
        eps = eps[: args.episodes]
    if not eps:
        print("没有找到 episode", file=sys.stderr)
        return 1

    resize = None
    if args.resize:
        w, h = args.resize.lower().split("x")
        resize = (int(w), int(h))

    # 用第一条探出图像尺寸, 以便声明 features
    la, ra, lg, rg, head0, wrist0 = load_episode(eps[0])
    h_img = decode_bgr2rgb(head0[0], resize)
    w_img = decode_bgr2rgb(wrist0[0], resize)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES},
        "observation.images.head": {
            "dtype": "video", "shape": h_img.shape,
            "names": ["height", "width", "channel"]},
        "observation.images.wrist": {
            "dtype": "video", "shape": w_img.shape,
            "names": ["height", "width", "channel"]},
    }

    print(f"源      {args.src}")
    print(f"episode {len(eps)} 条")
    print(f"图像    head {h_img.shape}  wrist {w_img.shape}")
    print(f"输出    {args.repo_id}  (fps={args.fps})\n")

    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=features,
        root=str(args.out / args.repo_id) if args.out else None,
        robot_type=args.robot_type, use_videos=True,
    )

    n_left = n_right = n_amb = 0
    n_frames = 0
    mismatched = []

    for ep_path in eps:
        idx = int("".join(c for c in ep_path.stem if c.isdigit()))
        la, ra, lg, rg, head, wrist = load_episode(ep_path)

        # 单臂假设的自检: 左右关节必须完全相同。不成立说明这不是单臂本体,
        # 或平台行为变了 —— 此时静默取左侧会丢掉右臂的全部信息。
        if not np.allclose(la, ra, atol=1e-6):
            mismatched.append(idx)

        side, ambiguous = pick_gripper_side(lg, rg)
        grip = lg if side == "left" else rg
        n_left += side == "left"
        n_right += side == "right"
        n_amb += ambiguous

        state14 = np.concatenate([la, grip[:, None]], axis=1).astype(np.float32)
        n = len(state14)
        if n < 2:
            print(f"  episode{idx}: 只有 {n} 帧, 跳过")
            continue

        task = load_instruction(inst_dir, idx, args.instruction_key)

        # action[t] = state[t+1], 故最后一帧没有对应动作, 产出 n-1 帧
        for t in range(n - 1):
            ds.add_frame({
                "observation.state": state14[t],
                "action": state14[t + 1],
                "observation.images.head": decode_bgr2rgb(head[t], resize),
                "observation.images.wrist": decode_bgr2rgb(wrist[t], resize),
                "task": task,
            })
        ds.save_episode()
        n_frames += n - 1
        print(f"  episode{idx:<4d} {n - 1:4d} 帧  夹爪取 {side}"
              f"{'  ⚠️ 两侧均无变化' if ambiguous else ''}")

    print(f"\n完成: {len(eps)} 条 / {n_frames} 帧")
    print(f"夹爪来源: left {n_left} 条, right {n_right} 条")
    if n_amb:
        print(f"⚠️ {n_amb} 条两侧夹爪均无变化 —— 该维不含信息, 若是抓取任务需排查")
    if mismatched:
        print(f"🛑 {len(mismatched)} 条的左右关节不一致, 单臂假设不成立: {mismatched[:10]}")
        print("   这些数据的右臂信息被丢弃了, 必须先查清原因再使用")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
