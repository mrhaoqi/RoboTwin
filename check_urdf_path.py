#!/usr/bin/env python3

import sys
import os
import yaml

# Add the RoboTwin path to sys.path
sys.path.append('/home/jy/workspace/RoboTwin')

from envs.beat_block_hammer import beat_block_hammer

def check_urdf_path():
    """Check which URDF file is actually being used during task execution"""
    
    print("🔍 CHECKING URDF PATH DURING TASK EXECUTION")
    print("=" * 60)
    
    try:
        # Create task environment with the same config as demo_randomized_rm65b
        task_config = {
            "task_name": "beat_block_hammer",
            "render_freq": 0,  # No rendering for this test
            "episode_num": 1,
            "use_seed": True,
            "save_freq": 15,
            "embodiment": ["rm_Lifting_robot_65B_jaw_description", "single_arm"],
            "domain_randomization": {
                "random_background": False,
                "messy_table": False,
                "clean_background_rate": 1,
                "random_head_camera_dis": 0,
                "random_table_height": 0,
                "random_light": False,
                "crazy_random_light_rate": 0
            },
            "camera": {
                "head_camera_type": "L515",
                "wrist_camera_type": "D435",
                "collect_head_camera": True,
                "collect_wrist_camera": True
            },
            "data_setting": {
                "rgb": True,
                "depth": False,
                "pointcloud": False,
                "observer": False,
                "endpose": False,
                "qpos": True,
                "mesh_segmentation": False,
                "actor_segmentation": False
            }
        }
        
        # Process embodiment configuration (same as collect_data.py)
        embodiment_type = task_config["embodiment"]
        
        # Load embodiment types configuration
        embodiment_config_path = "/home/jy/workspace/RoboTwin/task_config/_embodiment_config.yml"
        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        def get_embodiment_file(embodiment_type):
            robot_file = _embodiment_types[embodiment_type]["file_path"]
            if robot_file is None:
                raise "missing embodiment files"
            return robot_file
        
        def get_embodiment_config(robot_file):
            config_path = os.path.join(robot_file, "config.yml")
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.load(f.read(), Loader=yaml.FullLoader)
        
        # Handle single arm configuration (length 2)
        if len(embodiment_type) == 2:
            left_robot_file = get_embodiment_file(embodiment_type[0])
            right_robot_file = get_embodiment_file(embodiment_type[0])
            dual_arm_embodied = "single_arm"
            primary_arm = "left"
        else:
            raise ValueError(f"Unsupported embodiment configuration: {embodiment_type}")
        
        # Load embodiment configs
        left_embodiment_config = get_embodiment_config(left_robot_file)
        right_embodiment_config = get_embodiment_config(right_robot_file)
        
        print(f"📁 EMBODIMENT CONFIGURATION:")
        print(f"   Embodiment type: {embodiment_type}")
        print(f"   Left robot file: {left_robot_file}")
        print(f"   Right robot file: {right_robot_file}")
        print(f"   Dual arm embodied: {dual_arm_embodied}")
        
        print(f"\n📄 URDF CONFIGURATION:")
        print(f"   Left URDF path from config: {left_embodiment_config.get('urdf_path', 'NOT FOUND')}")
        print(f"   Right URDF path from config: {right_embodiment_config.get('urdf_path', 'NOT FOUND')}")
        
        # Calculate full URDF paths
        left_full_urdf_path = os.path.join(left_robot_file, left_embodiment_config.get('urdf_path', ''))
        right_full_urdf_path = os.path.join(right_robot_file, right_embodiment_config.get('urdf_path', ''))
        
        print(f"\n🎯 FULL URDF PATHS:")
        print(f"   Left full URDF path: {left_full_urdf_path}")
        print(f"   Right full URDF path: {right_full_urdf_path}")
        
        # Check if files exist
        print(f"\n✅ FILE EXISTENCE CHECK:")
        print(f"   Left URDF exists: {os.path.exists(left_full_urdf_path)}")
        print(f"   Right URDF exists: {os.path.exists(right_full_urdf_path)}")
        
        if os.path.exists(left_full_urdf_path):
            print(f"   Left URDF file size: {os.path.getsize(left_full_urdf_path)} bytes")
        if os.path.exists(right_full_urdf_path):
            print(f"   Right URDF file size: {os.path.getsize(right_full_urdf_path)} bytes")
        
        # Check what files are available in the urdf directory
        urdf_dir = os.path.join(left_robot_file, "urdf")
        if os.path.exists(urdf_dir):
            urdf_files = [f for f in os.listdir(urdf_dir) if f.endswith('.urdf')]
            print(f"\n📂 AVAILABLE URDF FILES IN {urdf_dir}:")
            for i, urdf_file in enumerate(urdf_files, 1):
                full_path = os.path.join(urdf_dir, urdf_file)
                size = os.path.getsize(full_path)
                print(f"   {i}. {urdf_file} ({size} bytes)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking URDF path: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_urdf_path()
    exit(0 if success else 1)
