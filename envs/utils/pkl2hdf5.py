import h5py, pickle
import numpy as np
import os
import cv2
from collections.abc import Mapping, Sequence
import shutil
from .images_to_video import images_to_video


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


def parse_dict_structure(data):
    if isinstance(data, dict):
        parsed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                parsed[key] = parse_dict_structure(value)
            elif isinstance(value, np.ndarray):
                parsed[key] = []
            else:
                parsed[key] = []
        return parsed
    else:
        return []


def append_data_to_structure(data_structure, data):
    for key in data_structure:
        if key in data:
            if isinstance(data_structure[key], list):
                # 如果是叶子节点，直接追加数据
                data_structure[key].append(data[key])
            elif isinstance(data_structure[key], dict):
                # 如果是嵌套字典，递归处理
                append_data_to_structure(data_structure[key], data[key])


def load_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_multi_camera_video(camera_data, video_path):
    """
    创建包含多个相机视角的视频

    Args:
        camera_data (dict): 相机名称到图像数组的映射
        video_path (str): 输出视频路径
    """
    if not camera_data:
        print("错误: 没有相机数据可用于生成视频")
        return

    # 获取相机名称列表，确保顺序一致
    camera_names = sorted(camera_data.keys())
    print(f"生成多相机视频，包含相机: {camera_names}")

    # 检查数据格式并处理
    decoded_camera_data = {}
    for camera_name in camera_names:
        frames_data = camera_data[camera_name]

        # 检查数据格式
        if len(frames_data) == 0:
            continue

        first_frame = frames_data[0]

        # 如果是numpy数组（原始RGB数据），直接使用
        if isinstance(first_frame, np.ndarray):
            decoded_camera_data[camera_name] = frames_data
            print(f"{camera_name}: 使用原始RGB数据，{len(frames_data)} 帧")

        # 如果是字节数据（JPEG编码），需要解码
        elif isinstance(first_frame, bytes):
            decoded_frames = []
            for encoded_frame in frames_data:
                # 移除padding的零字节
                encoded_frame = encoded_frame.rstrip(b'\0')
                # 解码图像
                nparr = np.frombuffer(encoded_frame, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    # OpenCV使用BGR，转换为RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    decoded_frames.append(img)

            if decoded_frames:
                decoded_camera_data[camera_name] = np.array(decoded_frames)
                print(f"{camera_name}: 解码了 {len(decoded_frames)} 帧")
        else:
            print(f"警告: {camera_name} 的数据格式不支持: {type(first_frame)}")

    if not decoded_camera_data:
        print("错误: 无法处理任何相机数据")
        return

    # 如果只有一个相机，直接生成视频
    if len(decoded_camera_data) == 1:
        single_camera_data = list(decoded_camera_data.values())[0]
        images_to_video(single_camera_data, out_path=video_path)
        return

    # 多相机情况：将相机视角水平拼接
    camera_names = sorted(decoded_camera_data.keys())
    first_camera_data = decoded_camera_data[camera_names[0]]
    n_frames = first_camera_data.shape[0]

    # 检查所有相机的帧数是否一致
    for camera_name in camera_names:
        camera_frames = decoded_camera_data[camera_name]
        if camera_frames.shape[0] != n_frames:
            print(f"警告: {camera_name} 的帧数 ({camera_frames.shape[0]}) 与其他相机不一致")
            n_frames = min(n_frames, camera_frames.shape[0])

    # 找到最大的高度，用于统一图像尺寸
    max_height = max(decoded_camera_data[name].shape[1] for name in camera_names)
    print(f"统一图像高度为: {max_height}")

    combined_frames = []
    for frame_idx in range(n_frames):
        frame_images = []
        for camera_name in camera_names:
            camera_frame = decoded_camera_data[camera_name][frame_idx]

            # 如果高度不匹配，调整图像尺寸
            if camera_frame.shape[0] != max_height:
                # 保持宽高比，调整到目标高度
                current_height, current_width = camera_frame.shape[:2]
                new_width = int(current_width * max_height / current_height)
                camera_frame = cv2.resize(camera_frame, (new_width, max_height))

            frame_images.append(camera_frame)

        # 水平拼接所有相机的图像
        combined_frame = np.concatenate(frame_images, axis=1)
        combined_frames.append(combined_frame)

    # 转换为numpy数组并生成视频
    combined_frames = np.array(combined_frames)
    images_to_video(combined_frames, out_path=video_path)

    print(f"多相机视频已保存: {video_path}")
    print(f"视频包含 {len(camera_names)} 个相机视角: {', '.join(camera_names)}")


def create_hdf5_from_dict(hdf5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            create_hdf5_from_dict(subgroup, value)
        elif isinstance(value, list):
            value = np.array(value)
            if "rgb" in key:
                encode_data, max_len = images_encoding(value)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
            else:
                hdf5_group.create_dataset(key, data=value)
        else:
            return
            try:
                hdf5_group.create_dataset(key, data=str(value))
                print("Not np array")
            except Exception as e:
                print(f"Error storing value for key '{key}': {e}")


def pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path):
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)

    # 收集所有可用的相机数据
    camera_data = {}
    observation = data_list.get("observation", {})

    # 检查所有可能的相机
    camera_names = ["head_camera", "left_camera", "right_camera"]
    for camera_name in camera_names:
        if camera_name in observation and "rgb" in observation[camera_name]:
            camera_rgb_data = observation[camera_name]["rgb"]
            if len(camera_rgb_data) > 0:  # 确保有数据
                camera_data[camera_name] = np.array(camera_rgb_data)

    if not camera_data:
        print("警告: 没有找到任何相机的RGB数据")
        return

    # 生成多相机视频
    create_multi_camera_video(camera_data, video_path)

    with h5py.File(hdf5_path, "w") as f:
        create_hdf5_from_dict(f, data_list)


def process_folder_to_hdf5_video(folder_path, hdf5_path, video_path):
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path)
