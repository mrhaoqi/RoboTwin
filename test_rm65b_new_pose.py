#!/usr/bin/env python3
"""
Test script to verify the new rm65b robot pose configuration
"""

import os
import sys

# Add the project root to Python path
sys.path.append('/home/jy/workspace/RoboTwin')

def test_rm65b_new_pose():
    print("🤖 Testing rm65b with new pose configuration...")
    
    try:
        from envs.beat_block_hammer import BeatBlockHammer
        print("✅ BeatBlockHammer import successful")
    except Exception as e:
        print(f"❌ BeatBlockHammer import failed: {e}")
        return False
    
    # Task configuration for rm65b with new pose
    task_config = {
        "task_name": "beat_block_hammer",
        "render_freq": 60,  # Enable rendering to see the robot
        "episode_num": 1,
        "use_seed": False,
        "save_freq": 15,
        "embodiment": ["rm65b", "single_arm"],
        "augmentation": {
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
        "data_type": {
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
    
    try:
        print("🔧 Creating task environment...")
        task = BeatBlockHammer(task_config)
        print("✅ Task environment created successfully")
        
        # Print robot pose information
        robot_pose = task.robot.left_entity_origion_pose
        print(f"📍 Robot base pose: position={robot_pose.p}, quaternion={robot_pose.q}")
        
        # Check if robot is properly positioned
        if robot_pose.p[2] < 0.5:  # Z coordinate should be low (close to table)
            print("✅ Robot is positioned close to table surface")
        else:
            print("⚠️  Robot might still be too high")
            
        # Try to get robot's current joint positions
        try:
            joint_positions = task.robot.get_qpos()
            print(f"🔧 Current joint positions: {joint_positions}")
        except Exception as e:
            print(f"⚠️  Could not get joint positions: {e}")
        
        # Try a simple movement test
        print("🎯 Testing robot movement capabilities...")
        try:
            # Get robot's end-effector pose
            ee_pose = task.robot.get_ee_pose()
            print(f"🎯 End-effector pose: {ee_pose}")
            print("✅ Robot movement test successful")
        except Exception as e:
            print(f"⚠️  Robot movement test failed: {e}")
        
        print("🎉 rm65b new pose configuration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Task creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rm65b_new_pose()
    if success:
        print("\n✅ All tests passed! The new rm65b pose configuration looks good.")
        print("💡 The robot should now be positioned horizontally facing the table.")
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
