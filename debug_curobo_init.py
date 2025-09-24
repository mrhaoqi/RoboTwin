#!/usr/bin/env python3
"""
Debug script to test CuRobo initialization for rm65b robot
"""

import os
import sys
import yaml
import traceback

# Add the project root to Python path
sys.path.append('/home/jy/workspace/RoboTwin')

def test_curobo_init():
    print("🔧 Testing CuRobo initialization for rm65b robot...")
    
    try:
        from envs.robot.planner import CuroboPlanner
        print("✅ CuroboPlanner import successful")
    except Exception as e:
        print(f"❌ CuroboPlanner import failed: {e}")
        traceback.print_exc()
        return False
    
    # Test configuration files
    config_files = [
        "assets/embodiments/rm_Lifting_robot_65B_jaw_description/curobo_left.yml",
        "assets/embodiments/rm_Lifting_robot_65B_jaw_description/curobo_right.yml"
    ]
    
    for config_file in config_files:
        print(f"\n🔍 Testing {config_file}...")
        
        if not os.path.exists(config_file):
            print(f"❌ Config file not found: {config_file}")
            continue
            
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Config file loaded successfully")
            
            # Check key configuration
            base_link = config['robot_cfg']['kinematics']['base_link']
            ee_link = config['robot_cfg']['kinematics']['ee_link']
            joint_names = config['robot_cfg']['kinematics']['cspace']['joint_names']
            retract_config = config['robot_cfg']['kinematics']['cspace']['retract_config']
            
            print(f"   Base Link: {base_link}")
            print(f"   EE Link: {ee_link}")
            print(f"   Joint Names: {joint_names}")
            print(f"   Retract Config: {retract_config}")
            print(f"   Joint Count: {len(joint_names)}")
            print(f"   Retract Count: {len(retract_config)}")
            
            if len(joint_names) != len(retract_config):
                print(f"❌ Mismatch: {len(joint_names)} joints vs {len(retract_config)} retract values")
            else:
                print(f"✅ Joint/retract config match")
                
        except Exception as e:
            print(f"❌ Config file error: {e}")
            traceback.print_exc()
    
    # Test URDF file
    urdf_path = "assets/embodiments/rm_Lifting_robot_65B_jaw_description/urdf/rm_Lifting_robot_65B_jaw_description_simplified_arm_only.urdf"
    print(f"\n🔍 Testing URDF file: {urdf_path}")
    
    if os.path.exists(urdf_path):
        print(f"✅ URDF file exists")
        try:
            with open(urdf_path, 'r') as f:
                urdf_content = f.read()
            
            # Check for key links
            key_links = ['footprint', 'arm_base_link', 'r_link6']
            for link in key_links:
                if f'<link name="{link}"' in urdf_content:
                    print(f"✅ Found link: {link}")
                else:
                    print(f"❌ Missing link: {link}")
                    
        except Exception as e:
            print(f"❌ URDF file error: {e}")
    else:
        print(f"❌ URDF file not found")
    
    # Test actual CuRobo initialization
    print(f"\n🚀 Testing actual CuRobo initialization...")

    try:
        from sapien.core import Pose

        # Create a dummy pose for testing
        dummy_pose = Pose([0, -0.65, 0], [0.707, 0, 0, 0.707])

        # Test left planner initialization
        config_path = os.path.abspath("assets/embodiments/rm_Lifting_robot_65B_jaw_description/curobo_left.yml")
        print(f"   Config path: {config_path}")

        active_joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', '4C2_Joint1', '4C2_Joint2', '4C2_Joint3', '4C2_Joint4']
        arm_joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

        print(f"   Active joints: {active_joints}")
        print(f"   Arm joints: {arm_joints}")

        planner = CuroboPlanner(
            robot_origion_pose=dummy_pose,
            active_joints_name=arm_joints,
            all_joints=active_joints,
            yml_path=config_path
        )

        print("✅ CuRobo planner initialized successfully!")

    except Exception as e:
        print(f"❌ CuRobo initialization failed: {e}")
        traceback.print_exc()

    # Test Robot class initialization
    print(f"\n🤖 Testing Robot class initialization...")

    try:
        import sapien.core as sapien
        from envs.robot.robot import Robot

        # Create a minimal scene
        engine = sapien.Engine()
        scene = engine.create_scene()

        # Create minimal kwargs for Robot initialization
        left_embodiment_config = {
            "urdf_path": "./urdf/rm_Lifting_robot_65B_jaw_description_fixed_platform.urdf",
            "move_group": ["r_link6", "r_link6"],
            "ee_joints": ["joint6", "joint6"],
            "arm_joints_name": [["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"], ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]],
            "gripper_name": [{"base": "4C2_Joint1", "mimic": [["4C2_Joint2", 1.0, 0.0]]}, {"base": "4C2_Joint1", "mimic": [["4C2_Joint2", 1.0, 0.0]]}],
            "gripper_bias": 0.03,
            "gripper_scale": 1.0,
            "homestate": [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            "robot_pose": [[0, -0.65, 0.0, 0.707, 0, 0, 0.707]]
        }

        kwargs = {
            "left_embodiment_config": left_embodiment_config,
            "right_embodiment_config": left_embodiment_config,
            "left_robot_file": "assets/embodiments/rm_Lifting_robot_65B_jaw_description",
            "right_robot_file": "assets/embodiments/rm_Lifting_robot_65B_jaw_description",
            "dual_arm_embodied": "single_arm",
            "primary_arm": "left"
        }

        print("   Creating Robot instance...")
        robot = Robot(scene, need_topp=False, **kwargs)
        print("✅ Robot instance created successfully!")

        print("   Calling set_planner...")
        robot.set_planner(scene)
        print("✅ set_planner completed!")

        print("   Checking planner attributes...")
        if hasattr(robot, 'left_planner'):
            print(f"✅ left_planner exists: {type(robot.left_planner)}")
        else:
            print("❌ left_planner missing")

        if hasattr(robot, 'right_planner'):
            print(f"✅ right_planner exists: {type(robot.right_planner)}")
        else:
            print("❌ right_planner missing")

    except Exception as e:
        print(f"❌ Robot initialization failed: {e}")
        traceback.print_exc()

    print("\n🎯 Debug complete!")

if __name__ == "__main__":
    test_curobo_init()
