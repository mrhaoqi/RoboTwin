# 导入系统模块，用于访问Python解释器变量和函数
import sys

# 将当前目录添加到Python模块搜索路径，确保可以导入本地模块
sys.path.append("./")

# 导入SAPIEN物理引擎核心模块，用于模拟物理环境
import sapien.core as sapien
# 导入SAPIEN渲染清理缓存函数，用于释放GPU内存
from sapien.render import clear_cache
# 导入有序字典类，保持键值对的插入顺序
from collections import OrderedDict
# 导入Python调试器，用于代码调试
import pdb
# 导入所有环境模块，包括各种任务环境类
from envs import *
# 导入YAML解析库，用于读取配置文件
import yaml
# 导入动态导入模块的库，用于运行时加载模块
import importlib
# 导入JSON处理库，用于处理JSON格式数据
import json
# 导入异常追踪模块，用于详细记录异常信息
import traceback
# 导入操作系统接口模块，用于文件和路径操作
import os
# 导入时间相关功能，用于延时和计时
import time
# 导入命令行参数解析器，用于处理命令行输入
from argparse import ArgumentParser

# 获取当前脚本文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录路径
parent_directory = os.path.dirname(current_file_path)

""" 根据任务名称动态导入并实例化对应的环境类。
该函数通过任务名称从envs包中导入相应模块，并实例化与任务名称同名的环境类。 如果找不到对应的任务类，则终止程序执行。
Args: task_name (str): 任务名称，用于定位和实例化对应的环境类。 该名称应与envs包中的模块名和类名一致。
Returns: object: 实例化的环境类对象。
Raises: SystemExit: 当找不到指定的任务类时抛出，并显示"No such task"错误信息。 """
def class_decorator(task_name):
    # 动态导入与任务名称对应的环境模块，格式为"envs.任务名称"
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        # 从导入的模块中获取与任务名称同名的类
        env_class = getattr(envs_module, task_name)
        # 实例化环境类，创建一个环境对象
        env_instance = env_class()
    except:
        # 如果找不到对应的任务类，抛出系统退出异常并显示错误信息
        raise SystemExit("No such task")
    # 返回实例化的环境对象
    return env_instance

""" 从机器人配置文件中加载机器人实施配置信息。
该函数读取指定机器人文件路径下的config.yml文件，并将其解析为Python对象。 配置文件包含机器人实施(embodiment)所需的各种参数。
Args: robot_file (str): 机器人文件夹的路径，该文件夹应包含config.yml配置文件
Returns: dict: 包含机器人实施配置参数的字典对象 """
def get_embodiment_config(robot_file):
    # 构建机器人配置文件的完整路径，指向robot_file目录下的config.yml文件
    robot_config_file = os.path.join(robot_file, "config.yml")
    # 以UTF-8编码打开配置文件进行读取
    with open(robot_config_file, "r", encoding="utf-8") as f:
        # 使用YAML加载器解析配置文件内容，转换为Python字典对象
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    # 返回解析后的配置参数字典
    return embodiment_args

""" 初始化并配置机器人任务环境，处理实施(embodiment)配置，并启动数据收集流程。
该函数负责加载任务配置，设置机器人实施参数，配置环境随机化选项，并启动数据收集过程。 
它处理单臂和双臂机器人配置，并显示当前配置信息。
Args: task_name (str): 任务名称，
用于加载对应的环境类 task_config (str): 任务配置文件名称（不含扩展名），
用于从task_config目录加载配置
Returns: None: 函数通过调用run()函数执行任务，不直接返回值
Raises: Exception: 当实施配置参数数量不是1或3时抛出异常 Exception: 当找不到实施文件时抛出异常 """
def main(task_name=None, task_config=None):
    # 使用任务名称动态创建对应的环境类实例
    task = class_decorator(task_name)
    # 构建任务配置文件的路径
    config_path = f"./task_config/{task_config}.yml"

    # 打开并读取任务配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        # 解析YAML格式的配置文件为Python字典
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    # 将任务名称添加到参数字典中，便于后续使用
    args['task_name'] = task_name

    # 从配置中获取实施(embodiment)类型信息
    embodiment_type = args.get("embodiment")
    # 构建实施配置文件的路径
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    # 打开并读取实施配置文件
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        # 解析YAML格式的实施配置为Python字典
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    # 定义内部函数，用于获取特定实施类型的文件路径
    def get_embodiment_file(embodiment_type):
        # 从实施类型配置中获取文件路径
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        # 如果文件路径为空，抛出异常
        if robot_file is None:
            raise "missing embodiment files"
        # 返回机器人文件路径
        return robot_file

    # 处理单臂机器人配置（实施类型长度为1）如：aloha-agilex是双臂机器人
    if len(embodiment_type) == 1:
        # 左右机械臂使用相同的机器人文件
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        # 设置为双臂实施模式
        args["dual_arm_embodied"] = True
    # 处理双臂机器人配置（实施类型长度为3，包含左臂、右臂和距离参数）如：franka-panda, franka-panda, 0.6
    elif len(embodiment_type) == 3:
        # 左右机械臂使用不同的机器人文件
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        # 设置机械臂之间的距离
        args["embodiment_dis"] = embodiment_type[2]
        # 设置为非双臂实施模式
        args["dual_arm_embodied"] = False
    else:
        # 如果实施类型参数数量不是1或3，抛出异常
        raise "number of embodiment config parameters should be 1 or 3"

    # 加载左机械臂的实施配置
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    # 加载右机械臂的实施配置
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    # 根据实施类型生成实施名称
    if len(embodiment_type) == 1:
        # 双臂机器人模式下，实施名称就是实施类型
        embodiment_name = str(embodiment_type[0])
    else:
        # 两个单臂机器人模式下，实施名称是左臂类型+右臂类型
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # 显示当前配置信息
    print("============= Config =============\n")
    # 显示是否使用杂乱桌面配置，使用紫色高亮显示标签
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    # 显示是否使用随机背景配置
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    # 如果启用随机背景，显示干净背景的比率
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    # 显示是否使用随机光照配置
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    # 如果启用随机光照，显示极端随机光照的比率
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    # 显示是否使用随机桌面高度配置
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    # 显示是否使用随机头部相机距离配置
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    # 显示头部相机配置信息，包括类型和是否收集数据，使用蓝色高亮显示标签
    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    # 显示手腕相机配置信息，包括类型和是否收集数据
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    # 显示实施配置名称
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    # 将实施名称添加到参数字典中
    args["embodiment_name"] = embodiment_name
    # 将任务配置名称添加到参数字典中
    args['task_config'] = task_config
    # 构建保存路径，包含任务名称和配置名称
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    # 调用run函数执行任务，传入任务环境和参数
    run(task, args)

""" 执行机器人任务环境的数据收集流程。
该函数实现了完整的数据收集过程，
分为两个主要阶段：种子收集和数据收集。 
在种子收集阶段，函数尝试多个随机种子，保存成功完成任务的种子ID； 
在数据收集阶段，使用这些成功的种子重新执行任务并收集详细数据。
Args: TASK_ENV: 任务环境实例，
提供任务执行的接口和方法 args (dict): 配置参数字典，
包含以下关键参数： 
  - domain_randomization (dict): 域随机化参数 
  - camera (dict): 相机参数 
  - save_path (str): 数据保存路径 
  - use_seed (bool): 是否使用已有种子文件 
  - episode_num (int): 需要收集的剧集数量 
  - render_freq (int): 渲染频率，0表示不渲染 
  - collect_data (bool): 是否执行数据收集阶段 
  - clear_cache_freq (int): 清理缓存的频率 
  - task_name (str): 当前任务名称 
  - task_config (str): 任务配置名称 
  - language_num (int): 生成指令的语言数量
Returns: None: 函数通过文件系统保存收集的数据，不直接返回值
Raises: UnStableError: 当任务执行过程中出现不稳定状态时抛出 AssertionError: 当数据收集过程中任务执行失败时抛出 """
def run(TASK_ENV, args):
    """执行任务环境的数据收集过程。
    
    该函数负责种子收集和数据收集两个主要阶段。首先收集成功的种子，然后基于这些种子收集实际数据。
    
    Args:
        TASK_ENV: 任务环境实例，提供任务执行的接口
        args (dict): 包含任务执行所需的各种参数的字典
    """
    # 初始化计数器和种子列表：
    # epid: 当前尝试的种子ID
    # suc_num: 成功收集的种子数量
    # fail_num: 失败的尝试次数
    # seed_list: 成功的种子ID列表
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    # 显示当前任务名称，使用蓝色高亮
    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== 收集种子阶段 ===========
    # 确保保存路径存在，如果不存在则创建
    os.makedirs(args["save_path"], exist_ok=True)

    # 如果不使用已有种子，则开始收集新的种子
    if not args["use_seed"]:
        # 显示开始收集种子的提示信息，使用黄色高亮
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        # 设置需要规划标志为True，表示需要进行动作规划
        args["need_plan"] = True

        # 检查是否存在已有的种子文件
        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            # 如果存在，读取种子文件内容
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                # 如果种子列表不为空，转换为整数并更新计数器
                if len(seed_list) != 0:
                    # 将字符串种子ID转换为整数
                    seed_list = [int(i) for i in seed_list]
                    # 更新成功收集的种子数量
                    suc_num = len(seed_list)
                    # 设置下一个种子ID为最大种子ID加1
                    epid = max(seed_list) + 1
            # 显示从已有种子文件继续的信息
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        # 循环收集种子，直到达到指定的剧集数量
        while suc_num < args["episode_num"]:
            try:
                # 设置演示环境，传入当前剧集编号、种子ID和其他参数
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                # 执行一次任务模拟
                TASK_ENV.play_once()

                # 检查规划是否成功且任务是否完成
                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    # 如果成功，打印成功信息
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    # 将成功的种子ID添加到列表中
                    seed_list.append(epid)
                    # 保存轨迹数据
                    TASK_ENV.save_traj_data(suc_num)
                    # 成功计数器加1
                    suc_num += 1
                else:
                    # 如果失败，打印失败信息
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    # 失败计数器加1
                    fail_num += 1

                # 关闭任务环境
                TASK_ENV.close_env()

                # 如果设置了渲染频率，关闭渲染器
                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                # 捕获不稳定错误异常
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                # 增加失败计数
                fail_num += 1
                # 关闭任务环境
                TASK_ENV.close_env()

                # 如果设置了渲染频率，关闭渲染器
                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                # 短暂延时，等待系统稳定
                time.sleep(0.3)
            except Exception as e:
                # 捕获其他所有异常
                # stack_trace = traceback.format_exc()  # 注释掉的堆栈跟踪代码
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                # 增加失败计数
                fail_num += 1
                # 关闭任务环境
                TASK_ENV.close_env()

                # 如果设置了渲染频率，关闭渲染器
                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                # 较长延时，等待系统恢复
                time.sleep(1)

            # 种子ID递增，准备下一次尝试
            epid += 1

            # 将当前成功的种子列表写入文件，实时更新
            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        # 完成种子收集，显示统计信息，失败次数使用红色高亮
        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        # 如果使用已有种子，显示提示信息，使用黄色高亮并居中显示
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        # 读取已有的种子文件
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            # 读取并分割种子列表
            seed_list = file.read().split()
            # 将字符串种子ID转换为整数
            seed_list = [int(i) for i in seed_list]

    # =========== 数据收集阶段 ===========

    # 如果启用数据收集
    if args["collect_data"]:
        # 显示开始数据收集的提示信息，使用黄色高亮
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        # 设置数据收集阶段的参数
        # 不需要规划，因为使用已有的轨迹数据
        args["need_plan"] = False
        # 关闭渲染，提高性能
        args["render_freq"] = 0
        # 启用数据保存
        args["save_data"] = True

        # 获取清理缓存的频率
        clear_cache_freq = args["clear_cache_freq"]

        # 初始化起始索引
        st_idx = 0

        # 定义内部函数，检查指定索引的HDF5文件是否存在
        def exist_hdf5(idx):
            # 构建HDF5文件路径
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            # 检查文件是否存在
            return os.path.exists(file_path)

        # 查找第一个不存在的HDF5文件索引，作为起始点
        # 这样可以避免覆盖已有的数据文件
        while exist_hdf5(st_idx):
            st_idx += 1

        # 从起始索引开始，循环处理每个剧集
        for episode_idx in range(st_idx, args["episode_num"]):
            # 显示当前任务名称，使用蓝色高亮
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            # 设置演示环境，使用对应索引的种子ID
            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

            # 加载该剧集的轨迹数据
            traj_data = TASK_ENV.load_tran_data(episode_idx)
            # 设置左右机械臂的关节路径
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            # 将路径列表设置到任务环境中
            TASK_ENV.set_path_lst(args)

            # 构建场景信息文件路径
            info_file_path = os.path.join(args["save_path"], "scene_info.json")

            # 如果场景信息文件不存在，创建一个空的JSON文件
            if not os.path.exists(info_file_path):
                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump({}, file, ensure_ascii=False)

            # 读取现有的场景信息数据库
            with open(info_file_path, "r", encoding="utf-8") as file:
                info_db = json.load(file)

            # 执行一次任务模拟，并获取场景信息
            info = TASK_ENV.play_once()
            # 将当前剧集的场景信息添加到数据库中
            info_db[f"episode_{episode_idx}"] = info

            # 将更新后的场景信息写回文件，使用美化格式
            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump(info_db, file, ensure_ascii=False, indent=4)

            # 关闭任务环境，根据清理缓存频率决定是否清理缓存
            TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            # 将PKL格式的数据合并为HDF5格式，并添加视频数据
            TASK_ENV.merge_pkl_to_hdf5_video()
            # 移除临时数据缓存
            TASK_ENV.remove_data_cache()
            # 断言检查任务是否成功完成，如果失败则抛出异常
            assert TASK_ENV.check_success(), "Collect Error"

        # 构建生成剧集指令的命令
        command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
        # 执行命令，生成剧集指令
        os.system(command)


# 当脚本作为主程序运行时执行的代码块
if __name__ == "__main__":
    # 导入渲染测试模块
    from test_render import Sapien_TEST
    # 执行SAPIEN渲染测试，确保渲染环境正常
    Sapien_TEST()

    # 导入PyTorch多进程模块
    import torch.multiprocessing as mp
    # 设置多进程启动方法为"spawn"，确保在多进程环境中的稳定性
    # force=True表示强制设置，即使已经设置过其他启动方法
    mp.set_start_method("spawn", force=True)

    # 创建命令行参数解析器
    parser = ArgumentParser()
    # 添加任务名称参数，必须提供
    parser.add_argument("task_name", type=str)
    # 添加任务配置参数，必须提供
    parser.add_argument("task_config", type=str)
    # 解析命令行参数
    parser = parser.parse_args()
    # 获取任务名称
    task_name = parser.task_name
    # 获取任务配置名称
    task_config = parser.task_config

    # 调用main函数，传入任务名称和配置名称，开始执行数据收集流程
    main(task_name=task_name, task_config=task_config)
