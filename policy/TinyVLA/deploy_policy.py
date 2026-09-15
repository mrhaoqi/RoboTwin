# import packages and module here
import os
import torch
import cv2
import time
import sys
import pickle
import numpy as np
# import torch_utils as TorchUtils
from torchvision import transforms
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from vla import *
from policy_heads import *
from aloha_scripts.constants import *
from data_utils.dataset import set_seed
from data_utils.robot_data_processor import InternVL3Process
from vla.model_load_utils import load_model_for_eval

def preprocess_img(images: torch.Tensor):
    assert images.ndim == 4 and images.shape[1] == 3
    original_size = (320, 240)
    new_size = (448, 448)
    ratio = 0.95
    t1 = transforms.Resize(size=original_size, antialias=True)
    t2 = transforms.Resize(size=new_size, antialias=True)
    images = t1(images)
    images = images[...,
             int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
             int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
    images = t2(images)

    return images
class TinyVLA:
    def __init__(self, policy_config, camera_names):
        super(TinyVLA).__init__()
        self.camera_names = camera_names
        self.policy_config = policy_config
        self.task_name = policy_config["task_name"]
        self.state_path = policy_config["state_path"]
        model_base = policy_config["model_base"] # if policy_config["enable_lore"] else None
        model_path = policy_config["model_path"]
        print("Start Load the Model")
        self.tokenizer, self.policy = load_model_for_eval(
            model_path=model_path,
            model_base=model_base,
            policy_config=policy_config
        )
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=False,attn_implementation="default")
        self.vla_process = InternVL3Process(
            tokenizer=self.tokenizer,
            conv_template=self.policy.conv_template,
            camera_names=self.camera_names,
            num_image_token=self.policy.num_image_token
        )
        with open(self.state_path, 'rb') as f:
            self.stats = pickle.load(f)


    def pre_process(self, sample):
        stats = self.stats
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(sample[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        qpos_data = torch.from_numpy(sample["qpos"]).float()
        qpos_data = (qpos_data - stats["qpos_mean"]) / stats["qpos_std"]
        qpos_data = qpos_data.unsqueeze(0)
        s = {
            'image': image_data,
            'state': qpos_data,
            'raw_lang': sample["raw_lang"],
        }
        return self.vla_process.preprocess(s)

    @torch.no_grad()
    def get_action(self, obs=None):
        # 必须禁用梯度：三路相机各切 12 个 448 图块（dynamic_preprocess 对 640x480 的结果），
        # 单步前向即有 36 个图块过视觉塔；若保留激活值用于反传，16 GB 显存会 OOM
        # （且未安装 FlashAttention，走 _naive_attn 时注意力矩阵的显存开销更大）。
        stats = self.stats
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
        # post_process = lambda a: a * stats['action_std'] + stats['action_mean']
        batch = self.pre_process(obs)
        # actions = self.policy.sample_action(**batch).detach().cpu().numpy()
        actions = self.policy.sample_action(**batch).detach().cpu().to(torch.float32).numpy()
        actions = np.squeeze(actions, axis=0)
        actions = post_process(actions)
        return actions


task_prompt = {
    "place_object_scale": "Use one arm to grab the object and put it on the scale.",
    "place_phone_stand": "Your task is to assist the robot in placing a phone onto a phone stand, both of which are randomly positioned on the desk at initialization. You will be provided with images of the desk from different angles to help determine the positions of the phone and phone stand, and to plan the necessary actions to accomplish the placement.",
    "blocks_stack_three": "Your task is to assist the robot in stacking three cubes on the desk in a specific order: red at the bottom, green in the middle, and blue on top. The cubes will be randomly placed on the desk at initialization. You will be provided with images from different angles to help determine the positions of the cubes and to plan the necessary actions to accomplish the stacking task.",
    "blocks_ranking_rgb": "Your task is to assist the robot in sorting three cubes on the desk so that they are arranged in the order of red, green, and blue from left to right. The cubes will be randomly placed on the desk at initialization. You will be provided with images from different angles to help determine the positions of the cubes and to plan the necessary actions to accomplish the sorting task.",
    "dual_shoes_place": "Your task is to assist the robot in placing two shoes into a shoe box, with the shoes oriented to the left. The shoes will be randomly placed on the floor or a surface at initialization, while the shoe box is fixed at a certain location. You will be provided with images from different angles to help determine the positions of the shoes and the shoe box, and to plan the necessary actions to accomplish the task.",
    "put_bottles_dustbin": "Your task is to assist the robot in putting three bottles into the trash bin. The bottles are randomly placed on the desk at initialization. You will be provided with images of the desk from different angles to help determine the positions of the bottles and the trash bin, and to plan the necessary actions to accomplish the task.",
}

def encode_obs(observation):  # Post-Process Observation
    """
    Process input data for VLA model。
    """
    obs = observation
    cam_high = obs["observation"]["head_camera"]["rgb"]
    cam_left = obs["observation"]["left_camera"]["rgb"]
    cam_right = obs["observation"]["right_camera"]["rgb"]
    # 必须与训练数据保持相同的 640x480 尺寸，不可预先压成 448x448 方图。
    # InternVL3Process.load_image 内部的 dynamic_preprocess 会按宽高比动态切图块：
    #   640x480 (4:3) -> 12 个 448 图块；448x448 (1:1) -> 仅 1 个图块。
    # 若此处先 resize 成方图，模型收到的视觉 token 数量与训练时相差 12 倍。
    # 训练侧（data_utils/dataset.py）直接使用 HDF5 中的 640x480 原图，不做缩放。
    cam_right = cv2.resize(cam_right, (640, 480))
    cam_left = cv2.resize(cam_left, (640, 480))
    cam_high = cv2.resize(cam_high, (640, 480))
    # 7 维单臂格式（环境变量 TINYVLA_ACTION_DIM=7 启用）：
    # 单臂本体下左右两半的 6 个关节角完全相同（读的是同一批 joint 对象），
    # 故只取一侧；夹爪同理取左侧 —— 观测中的两个夹爪值在单臂下由同一物理关节决定。
    # 训练数据经 convert_to_7dim.py 转换时，夹爪取的是"实际在动的那一侧"，
    # 而运行时该侧的值即为当前夹爪真实开合度，两者语义一致。
    if os.environ.get("TINYVLA_ACTION_DIM") == "7":
        qpos = (observation["joint_action"]["left_arm"] +
                [observation["joint_action"]["left_gripper"]])
    else:
        qpos = (observation["joint_action"]["left_arm"] + [observation["joint_action"]["left_gripper"]] +
                observation["joint_action"]["right_arm"] + [observation["joint_action"]["right_gripper"]])
    qpos = np.array(qpos)
    return {
        "cam_high": cam_high,
        "cam_left": cam_left,
        "cam_right": cam_right,
        "qpos": qpos,
    }


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    """
    Load Model.
    """
    action_head = 'unet_diffusion_policy'
    camera_names = ['cam_high', 'cam_left', 'cam_right']
    task_name = usr_args["task_name"]
    model_dir = usr_args["model_path"]
    model_base = usr_args["model_base"]
    state_path = usr_args["state_path"]
    policy_config = {
        "task_name": task_name,
        "model_path": model_dir,
        "model_base": model_base,
        "state_path": state_path,
        "enable_lora": False,
        "action_head": action_head,
    }
    model = TinyVLA(policy_config, camera_names)
    return model  # return your policy model


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)  # Post-Process Observation
    # instruction = task_prompt[model.task_name]
    instruction = TASK_ENV.get_instruction()
    obs.update({"raw_lang": str(instruction)})
    # print("******************************")
    actions = model.get_action(obs)  # Get Action according to observation chunk

    # 7 维单臂输出 -> 平台的 14 维双臂接口。
    # TASK_ENV.take_action 按 [左臂6, 左夹爪, 右臂6, 右夹爪] 拆分，故将同一条臂的
    # 6 个关节角与夹爪值复制到左右两路：单臂下二者本就驱动同一批物理关节
    # （见 envs/_base_task.py 中 is_single_arm_embodiment 的处理），复制后完全一致，
    # 不会再出现两路预测不同而相互覆盖的问题。
    if os.environ.get("TINYVLA_ACTION_DIM") == "7":
        actions = np.concatenate([actions, actions], axis=-1)

    for action in actions:  # Execute each step of the action
        # TASK_ENV.take_one_step_action(action)
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
    return observation


def reset_model(model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    pass
