#!/usr/bin/env python3

import sys
import os
import yaml
import numpy as np
import sapien.core as sapien
import time

# Add the RoboTwin path to sys.path
sys.path.append('/home/jy/workspace/RoboTwin')

from envs.beat_block_hammer import beat_block_hammer

def visualize_full_environment():
    """Visualize the complete environment with robot and task objects"""
    
    print("🎬 VISUALIZING COMPLETE ENVIRONMENT")
    print("=" * 60)
    
    try:
        # Create the task environment (this includes scene, robot, objects, etc.)
        print("🔧 Creating task environment...")
        
        # Task configuration for rm65b
        task_config = {
            "task_name": "beat_block_hammer",
            "render_freq": 60,  # Enable rendering
            "episode_num": 1,
            "use_seed": False,
            "save_freq": 15,
            "embodiment": ["rm65b", "single_arm"],  # Our single arm config
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
        
        # Create scene using modern Sapien API
        scene = sapien.Scene()
        scene.set_timestep(1 / 240.0)

        # Add better lighting
        scene.set_ambient_light([0.3, 0.3, 0.3])
        scene.add_directional_light([1, -1, -1], [1.0, 1.0, 1.0])
        scene.add_directional_light([-1, 1, -1], [0.5, 0.5, 0.5])

        # Create viewer
        viewer = scene.create_viewer()
        # Position camera to see robot arm (which extends from Y=-0.279 to Y=-0.854)
        # Place camera to look at the robot arm area
        viewer.set_camera_xyz(x=1.5, y=-0.5, z=1.2)  # Look at robot arm area
        viewer.set_camera_rpy(r=0, p=-0.3, y=0.5)     # Angle to see the arm

        print(f"📷 Camera positioned at: x=1.5, y=-0.5, z=1.2 (looking at robot arm)")
        print(f"📷 Camera rotation: r=0, p=-0.3, y=0.5")

        # Simulate the arguments that would be passed to Robot.__init__
        args = {
            "left_robot_file": "/home/jy/workspace/RoboTwin/assets/embodiments/rm65b",
            "right_robot_file": "/home/jy/workspace/RoboTwin/assets/embodiments/rm65b",
            "dual_arm_embodied": "single_arm",
            "primary_arm": "left"
        }

        # Load embodiment configs
        config_path = os.path.join(args["left_robot_file"], "config.yml")
        with open(config_path, "r", encoding="utf-8") as f:
            left_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        args["left_embodiment_config"] = left_config
        args["right_embodiment_config"] = left_config  # Same config for single arm

        # Create the robot
        from envs.robot.robot import Robot
        robot = Robot(scene, need_topp=False, **args)

        # Important: Call init_joints to properly initialize left_ee and other attributes
        robot.init_joints()

        # Move robot to homestate for better visibility
        homestate = left_config.get("homestate", [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        if robot.is_single_arm:
            target_qpos = homestate[0]  # Use left arm homestate
        else:
            target_qpos = homestate[0]  # Use left arm homestate

        # Set robot to homestate
        current_qpos = robot.left_entity.get_qpos()
        arm_dof = len(robot.left_arm_joints_name)
        new_qpos = current_qpos.copy()
        new_qpos[:arm_dof] = target_qpos
        robot.left_entity.set_qpos(new_qpos)

        print(f"🤖 Setting robot to homestate: {target_qpos}")
        print(f"🤖 Robot arm joints: {robot.left_arm_joints_name}")
        print(f"🤖 Total DOF: {len(current_qpos)}, Arm DOF: {arm_dof}")
        print(f"🤖 Robot base position: {robot.left_entity.get_pose().p}")
        print(f"🤖 URDF path being used: {robot.left_urdf_path}")
        print(f"🤖 Robot entity name: {robot.left_entity.get_name()}")

        # Print all links in the robot for debugging
        all_links = robot.left_entity.get_links()
        print(f"🤖 Robot has {len(all_links)} links:")

        # Find the arm links specifically
        arm_links = []
        for i, link in enumerate(all_links):
            link_name = link.get_name()
            link_pose = link.get_pose()
            if i < 15:  # Show first 15 links
                print(f"   {i+1:2d}. {link_name:25s} at [{link_pose.p[0]:6.3f}, {link_pose.p[1]:6.3f}, {link_pose.p[2]:6.3f}]")

            # Look for arm links
            if 'r_link' in link_name or 'joint' in link_name:
                arm_links.append((link_name, link_pose.p))

        print(f"\n🦾 ARM LINKS FOUND:")
        for link_name, pos in arm_links:
            print(f"   {link_name:25s} at [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")

        # Check ee joint initialization
        print(f"🤖 EE Joint Debug:")
        print(f"   left_ee_name: {robot.left_ee_name}")
        print(f"   left_ee object: {getattr(robot, 'left_ee', 'NOT FOUND')}")
        if hasattr(robot, 'left_ee') and robot.left_ee:
            print(f"   left_ee pose: {robot.left_ee.get_global_pose()}")

        # Print all joints for debugging
        all_joints = robot.left_entity.get_active_joints()
        print(f"🤖 Robot has {len(all_joints)} active joints:")
        for i, joint in enumerate(all_joints[:10]):  # Show first 10 joints
            joint_name = joint.get_name()
            print(f"   {i+1:2d}. {joint_name}")

        # Add a simple table for context (positioned relative to robot at [0, 0, 0])
        table_builder = scene.create_actor_builder()
        table_builder.add_box_collision(half_size=[0.4, 0.6, 0.02])
        table_builder.add_box_visual(half_size=[0.4, 0.6, 0.02])
        table = table_builder.build_static(name="table")
        table.set_pose(sapien.Pose([0.5, 0, 0.72]))  # Position table in front of robot

        # Add some simple objects for context
        # Hammer-like object
        hammer_builder = scene.create_actor_builder()
        hammer_builder.add_box_collision(half_size=[0.02, 0.02, 0.1])
        hammer_builder.add_box_visual(half_size=[0.02, 0.02, 0.1])
        hammer = hammer_builder.build_kinematic(name="hammer")
        hammer.set_pose(sapien.Pose([0.4, -0.1, 0.84]))  # On table

        # Block-like object
        block_builder = scene.create_actor_builder()
        block_builder.add_box_collision(half_size=[0.03, 0.03, 0.03])
        block_builder.add_box_visual(half_size=[0.03, 0.03, 0.03])
        block = block_builder.build_kinematic(name="block")
        block.set_pose(sapien.Pose([0.6, 0.1, 0.77]))  # On table

        # Add some visual markers to help locate the robot
        # Add coordinate frame markers
        for i, (name, pos, color) in enumerate([
            ("X-axis", [0.2, 0, 0], [1, 0, 0, 1]),
            ("Y-axis", [0, 0.2, 0], [0, 1, 0, 1]),
            ("Z-axis", [0, 0, 0.2], [0, 0, 1, 1])
        ]):
            marker_builder = scene.create_actor_builder()
            marker_builder.add_sphere_collision(radius=0.05)
            marker_builder.add_sphere_visual(radius=0.05, material=sapien.render.RenderMaterial(base_color=color))
            marker = marker_builder.build_static(name=f"marker_{name}")
            marker.set_pose(sapien.Pose(pos))

        # Add a large marker at ACTUAL robot base position
        try:
            robot_base_pos = robot.left_entity.get_root_pose().p
            print(f"🎯 Robot base position: {robot_base_pos}")
        except:
            robot_base_pos = [0, -0.4, 1.0]  # Use config position as fallback
            print(f"🎯 Using fallback robot base position: {robot_base_pos}")

        robot_marker_builder = scene.create_actor_builder()
        robot_marker_builder.add_sphere_collision(radius=0.1)
        robot_marker_builder.add_sphere_visual(radius=0.1, material=sapien.render.RenderMaterial(base_color=[1, 1, 0, 1]))  # Yellow
        robot_marker = robot_marker_builder.build_static(name="robot_base_marker")
        robot_marker.set_pose(sapien.Pose(robot_base_pos))  # At ACTUAL robot base

        # Add marker at ACTUAL end effector position
        try:
            ee_pose = robot.get_ee_pose()
            if isinstance(ee_pose, list) and len(ee_pose) >= 3:
                ee_pos = ee_pose[:3]
            else:
                ee_pos = [ee_pose.p[0], ee_pose.p[1], ee_pose.p[2]]
            print(f"🎯 EE position: {ee_pos}")
        except Exception as e:
            print(f"❌ Could not get EE pose: {e}")
            # Use the pose from debug output
            ee_pos = [0.847, -0.385, 0.146]
            print(f"🎯 Using fallback EE position: {ee_pos}")

        ee_marker_builder = scene.create_actor_builder()
        ee_marker_builder.add_sphere_collision(radius=0.08)
        ee_marker_builder.add_sphere_visual(radius=0.08, material=sapien.render.RenderMaterial(base_color=[1, 0, 1, 1]))  # Magenta
        ee_marker = ee_marker_builder.build_static(name="ee_marker")
        ee_marker.set_pose(sapien.Pose(ee_pos))  # At ACTUAL end effector position

        # Store references for later use
        task_env = type('MockTaskEnv', (), {
            'scene': scene,
            'robot': robot,
            'viewer': viewer,
            'hammer': hammer,
            'block': block,
            'table': table
        })()
        
        print("✅ Task environment created successfully!")
        
        # Print environment information
        print(f"\n🌍 ENVIRONMENT INFORMATION:")
        print(f"   Scene: {scene}")
        print(f"   Robot: {robot}")
        print(f"   Single arm mode: {robot.is_single_arm}")
        print(f"   Primary arm: {robot.primary_arm}")

        # Print camera information
        print(f"\n📷 CAMERA INFORMATION:")
        if hasattr(robot, 'left_camera') and robot.left_camera:
            print(f"   Left camera: {robot.left_camera.get_name()}")
        else:
            print(f"   Left camera: Not found")
        if hasattr(robot, 'right_camera') and robot.right_camera:
            print(f"   Right camera: {robot.right_camera.get_name()}")
        else:
            print(f"   Right camera: Not found")

        # Print task objects
        print(f"\n🔨 TASK OBJECTS:")
        print(f"   Hammer: {hammer}")
        print(f"   Block: {block}")
        print(f"   Table: {table}")
        
        # Print robot joint information
        print(f"\n🤖 ROBOT JOINT STATUS:")
        current_qpos = robot.left_entity.get_qpos()
        arm_qpos = current_qpos[:len(robot.left_arm_joints_name)]
        print(f"   Current positions: {[f'{q:6.3f}' for q in arm_qpos]}")
        print(f"   Target homestate:  {[f'{h:6.3f}' for h in homestate[0]]}")

        # Print all actors in the scene for debugging
        all_actors = scene.get_all_actors()
        print(f"\n🎭 ALL SCENE ACTORS ({len(all_actors)}):")
        for i, actor in enumerate(all_actors):
            actor_name = actor.get_name()
            actor_pose = actor.get_pose()
            print(f"   {i+1:2d}. {actor_name:30s} at [{actor_pose.p[0]:6.3f}, {actor_pose.p[1]:6.3f}, {actor_pose.p[2]:6.3f}]")

        # Get end effector pose
        try:
            if hasattr(robot, 'left_ee') and robot.left_ee:
                ee_pose = robot.get_left_ee_pose()
                if hasattr(ee_pose, 'p'):  # If it's a Pose object
                    print(f"\n🎯 END EFFECTOR STATUS:")
                    print(f"   Position: [{ee_pose.p[0]:6.3f}, {ee_pose.p[1]:6.3f}, {ee_pose.p[2]:6.3f}]")
                    print(f"   Quaternion: [{ee_pose.q[0]:6.3f}, {ee_pose.q[1]:6.3f}, {ee_pose.q[2]:6.3f}, {ee_pose.q[3]:6.3f}]")
                else:  # If it's a list
                    print(f"\n🎯 END EFFECTOR STATUS:")
                    print(f"   Pose (list): {[f'{x:6.3f}' for x in ee_pose]}")
            else:
                print(f"\n🎯 END EFFECTOR STATUS:")
                print(f"   ❌ End effector joint '{robot.left_ee_name}' not found in robot")
        except Exception as e:
            print(f"   ❌ Error getting end effector pose: {e}")
        
        # Interactive visualization
        print(f"\n" + "=" * 60)
        print(f"🎬 INTERACTIVE VISUALIZATION")
        print(f"   The complete environment is now displayed.")
        print(f"   You can see:")
        print(f"   - The rm65b robot in its current pose")
        print(f"   - The table and task objects (hammer, block)")
        print(f"   - The complete scene setup")
        print(f"   ")
        print(f"   Controls:")
        print(f"   - Mouse: Rotate view")
        print(f"   - WASD: Move camera")
        print(f"   - ESC: Exit")
        print(f"=" * 60)
        
        # Keep the viewer open
        step_count = 0
        while not viewer.closed:
            scene.step()
            viewer.render()

            # Print periodic updates
            step_count += 1
            if step_count % 240 == 0:  # Every 4 seconds at 60 FPS
                current_qpos = robot.left_entity.get_qpos()
                arm_qpos = current_qpos[:len(robot.left_arm_joints_name)]
                print(f"   Step {step_count:4d}: Joint positions {[f'{q:6.3f}' for q in arm_qpos]}")

                # Get end effector pose for debugging
                try:
                    ee_pose = robot.get_left_ee_pose()
                    if hasattr(ee_pose, 'p'):
                        print(f"   EE Position: [{ee_pose.p[0]:6.3f}, {ee_pose.p[1]:6.3f}, {ee_pose.p[2]:6.3f}]")
                    else:
                        print(f"   EE Pose: {[f'{x:6.3f}' for x in ee_pose[:3]]}")
                except Exception as e:
                    print(f"   EE Error: {e}")

            time.sleep(1/60)  # 60 FPS
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create or visualize environment: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if 'viewer' in locals():
            try:
                viewer.close()
            except:
                pass

if __name__ == "__main__":
    success = visualize_full_environment()
    exit(0 if success else 1)
