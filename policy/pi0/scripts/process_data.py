import sys

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import yaml, json


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def get_task_config(task_name):
    with open(f"./task_config/{task_name}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return args


# ============ 单臂本体数据规整 ============
# RoboTwin 的「单臂模式」（embodiment 列表长度为 1）实为 dual_arm_embodied=True，
# 同一台机器人同时充当左右臂，数据仍是 14 维双臂格式。任务按物体位置动态选臂
# （如 envs/place_object_stand.py:106），导致有效夹爪信号在 state[6] 与 state[13]
# 之间逐 episode 翻转，未使用侧恒为初值 1.0（死值）。
# 实测 place_object_stand 10 条：6 条 left、4 条 right。
# 规整做法：把有效夹爪值同时写入两个位置，臂关节数据不动、维度不变。
NORMALIZE_GRIPPER = True

_GRIPPER_ACTIVE_EPS = 0.05  # max-min 超过此值视为该通道有动作


def load_scene_info(path):
    """读取 scene_info.json；不存在时返回空 dict（由启发式兜底）。"""
    p = os.path.join(path, "scene_info.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[normalize] 读取 {p} 失败（{e}），改用启发式判定")
        return {}


def resolve_active_arm(scene_info, ep_idx, left_gripper, right_gripper):
    """判定该 episode 的有效臂。

    返回 "left" / "right" / None（None 表示不做规整）。

    一级：scene_info.json 的 info["{a}"]，由任务代码写入，权威。
    二级：取两个夹爪通道中 max-min 较大者。

    两种情况明确不规整（返回 None）并告警，不做猜测：
      - 双通道都有动作 → 真双臂数据（piper / aloha 等），本规整不适用
      - 双通道都无动作 → 夹爪全程未动，无信号可传递
    """
    l_rng = float(np.max(left_gripper) - np.min(left_gripper))
    r_rng = float(np.max(right_gripper) - np.min(right_gripper))
    l_act, r_act = l_rng > _GRIPPER_ACTIVE_EPS, r_rng > _GRIPPER_ACTIVE_EPS

    if l_act and r_act:
        print(f"[normalize] episode{ep_idx}: 左右夹爪均有动作"
              f"（{l_rng:.3f} / {r_rng:.3f}），判定为真双臂数据，跳过规整")
        return None
    if not l_act and not r_act:
        print(f"[normalize] episode{ep_idx}: 左右夹爪均无动作"
              f"（{l_rng:.3f} / {r_rng:.3f}），无有效信号，跳过规整")
        return None

    tag = scene_info.get(f"episode_{ep_idx}", {}).get("info", {}).get("{a}")
    if tag in ("left", "right"):
        heuristic = "left" if l_rng > r_rng else "right"
        if tag != heuristic:
            print(f"[normalize] episode{ep_idx}: scene_info 标记为 {tag}，"
                  f"但夹爪方差指向 {heuristic}，以 scene_info 为准")
        return tag

    heuristic = "left" if l_rng > r_rng else "right"
    print(f"[normalize] episode{ep_idx}: scene_info 缺少 {{a}} 标记，"
          f"按夹爪方差判定为 {heuristic}")
    return heuristic


def data_transform(path, episode_num, save_path, normalize_gripper=NORMALIZE_GRIPPER):
    begin = 0
    floders = os.listdir(path)
    scene_info = load_scene_info(path) if normalize_gripper else {}
    # assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):

        desc_type = "seen"
        instruction_data_path = os.path.join(path, "instructions", f"episode{i}.json")
        with open(instruction_data_path, "r") as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict[desc_type]
        save_instructions_json = {"instructions": instructions}

        os.makedirs(os.path.join(save_path, f"episode_{i}"), exist_ok=True)

        with open(
                os.path.join(os.path.join(save_path, f"episode_{i}"), "instructions.json"),
                "w",
        ) as f:
            json.dump(save_instructions_json, f, indent=2)

        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = (load_hdf5(
            os.path.join(path, "data", f"episode{i}.hdf5")))
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        active_arm = (resolve_active_arm(scene_info, i, left_gripper_all, right_gripper_all)
                      if normalize_gripper else None)

        last_state = None
        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            # 规整：把有效臂的夹爪值同时写入两个通道，消除逐 episode 的通道翻转。
            # active_arm 为 None 时保持上游原始行为。
            if active_arm == "left":
                right_gripper = left_gripper
            elif active_arm == "right":
                left_gripper = right_gripper

            state = np.array(left_arm.tolist() + [left_gripper] + right_arm.tolist() + [right_gripper])  # joints angle

            state = state.astype(np.float32)

            if j != left_gripper_all.shape[0] - 1:
                qpos.append(state)

                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0:
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        hdf5path = os.path.join(save_path, f"episode_{i}/episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")
            image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        default="beat_block_hammer",
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("setting", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        default=50,
        help="Number of episodes to process (e.g., 50)",
    )
    parser.add_argument(
        "--no-normalize-gripper",
        action="store_true",
        help="关闭单臂本体的夹爪通道规整，保持上游原始行为（用于对照与回归）",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="覆盖默认输出目录（默认 processed_data/<task>-<setting>-<num>）",
    )
    args = parser.parse_args()

    task_name = args.task_name
    setting = args.setting
    expert_data_num = args.expert_data_num

    load_dir = os.path.join("../../data", str(task_name), str(setting))

    begin = 0
    print(f'read data from path:{os.path.join("data", load_dir)}')

    target_dir = args.save_dir or f"processed_data/{task_name}-{setting}-{expert_data_num}"
    normalize = not args.no_normalize_gripper
    print(f"[normalize] 夹爪通道规整: {'开启' if normalize else '关闭'}")
    begin = data_transform(
        load_dir,
        expert_data_num,
        target_dir,
        normalize_gripper=normalize,
    )
