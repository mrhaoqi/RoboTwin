DATA_DIR = "/data/robotiwin/policy/TinyVLA/data"
PRETRAIN_DIR = '/data/h5py2'
LOCAL_DATA_DIR = '/home/data'

TASK_CONFIGS = {
    "task_name_you_test": {
        'dataset_dir': [DATA_DIR + "/sim-/aloha-agilex-1-m1_b1_l1_h0.03_c0_D435-100"],
        'episode_len': 500,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        "sample_weights": [1, 1]
    },
    # 2026-08-14: rm65b 本体 place_object_stand，50 条 episode（每条约 99 帧）
    "place_object_stand": {
        'dataset_dir': ["data/sim-place_object_stand/demo_rm65b_single-50"],
        'episode_len': 200,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        "sample_weights": [1]
    },
    # 2026-08-18: rm65b(桌面装) + rm65b_lateral(侧装 y=-0.70 z=1.05) 混合，各 50 条。
    # 让策略同时见到两种安装姿态；两批数据的域随机化参数一致，故等权采样。
    "place_object_stand_mix": {
        'dataset_dir': [
            "data/sim-place_object_stand/demo_rm65b_single-50",
            "data/sim-place_object_stand/demo_rm65b_lateral_50-50",
        ],
        'episode_len': 200,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        "sample_weights": [1, 1]
    },
    # 2026-08-20: 数据量扩充至每构型 150 条（共 300），用于验证
    # 「数据量不足导致欠拟合」这一判断（详见实验报告 8.6 / 8.7 节）。
    "place_object_stand_mix300": {
        'dataset_dir': [
            "data/sim-place_object_stand/demo_rm65b_single_150-150",
            "data/sim-place_object_stand/demo_rm65b_lateral_150-150",
        ],
        'episode_len': 200,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        "sample_weights": [1, 1]
    },
    "turn_switch_mix300": {
        'dataset_dir': [
            "data/sim-turn_switch/demo_rm65b_single_150-150",
            "data/sim-turn_switch/demo_rm65b_lateral_150-150",
        ],
        'episode_len': 200,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        "sample_weights": [1, 1]
    },
    # 2026-08-21: 7 维单臂格式（6 关节 + 1 夹爪），由 14 维数据经 convert_to_7dim.py 转换。
    # 消除双臂格式对单臂本体造成的冗余：14 维中左右两半的关节角完全相同（差值恒为 0），
    # 且模型需额外学习"本次驱动哪一侧夹爪"这一本不存在的判别任务。
    # 训练时须同步设置 --action_dim 7 --state_dim 7。
    "place_object_stand_mix300_7dim": {
        'dataset_dir': [
            "data/sim-place_object_stand/demo_rm65b_single_150-150_7dim",
            "data/sim-place_object_stand/demo_rm65b_lateral_150-150_7dim",
        ],
        'episode_len': 200,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        "sample_weights": [1, 1]
    },
}

### ALOHA fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
FPS = 50
# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / \
            (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (
            PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (
            MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (
            PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (
            MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (
            PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (
            MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (
            PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (
            MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (
            PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE) / 2
