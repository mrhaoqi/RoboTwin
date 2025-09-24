#!/usr/bin/env python3

import sys
import os
import yaml
import numpy as np
import sapien.core as sapien

# Add the RoboTwin path to sys.path
sys.path.append('/home/jy/workspace/RoboTwin')

from envs.robot.robot import Robot

def diagnose_eigenvalue_error():
    """Diagnose the 'Eigenvalues did not converge' error"""
    
    print("🔍 DIAGNOSING EIGENVALUE CONVERGENCE ERROR")
    print("=" * 60)
    
    # Create a simple Sapien scene
    scene = sapien.Scene()
    
    # Simulate the arguments that would be passed to Robot.__init__
    args = {
        "left_robot_file": "/home/jy/workspace/RoboTwin/assets/embodiments/rm_Lifting_robot_65B_jaw_description",
        "right_robot_file": "/home/jy/workspace/RoboTwin/assets/embodiments/rm_Lifting_robot_65B_jaw_description",
        "dual_arm_embodied": "single_arm",
        "primary_arm": "left"
    }
    
    # Load embodiment configs
    config_path = os.path.join(args["left_robot_file"], "config.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        left_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    args["left_embodiment_config"] = left_config
    args["right_embodiment_config"] = left_config  # Same config for single arm
    
    print(f"✓ Configuration loaded")
    
    try:
        # Create the robot
        robot = Robot(scene, **args)
        robot.set_planner(scene)
        robot.init_joints()  # Initialize joints - this is crucial!
        
        print(f"✅ Robot and planner created successfully!")
        
        # Check homestate configuration
        print(f"\n🏠 HOMESTATE ANALYSIS:")
        print(f"   Left homestate: {robot.left_homestate}")
        print(f"   Right homestate: {robot.right_homestate}")
        print(f"   Homestate length: {len(robot.left_homestate)}")
        print(f"   Expected joints: {len(robot.left_arm_joints_name)}")
        
        if len(robot.left_homestate) != len(robot.left_arm_joints_name):
            print(f"   ❌ HOMESTATE LENGTH MISMATCH!")
            print(f"   💡 Homestate has {len(robot.left_homestate)} values but {len(robot.left_arm_joints_name)} joints")
        else:
            print(f"   ✅ Homestate length matches joint count")
        
        # Check joint limits
        print(f"\n🔧 JOINT LIMITS ANALYSIS:")
        all_joints = robot.left_entity.get_active_joints()

        # Get arm joints by name
        arm_joint_names = robot.left_arm_joints_name
        arm_joints = []
        for joint_name in arm_joint_names:
            for joint in all_joints:
                if joint.get_name() == joint_name:
                    arm_joints.append(joint)
                    break
        
        print(f"   Total active joints: {len(all_joints)}")
        print(f"   Arm joints: {len(arm_joints)}")
        
        homestate_issues = []
        for i, (joint, target_angle) in enumerate(zip(arm_joints, robot.left_homestate)):
            joint_name = joint.get_name()
            joint_limits = joint.get_limits()

            print(f"   Joint {i+1} ({joint_name}):")

            # Handle different joint limit formats
            try:
                if len(joint_limits) == 2 and hasattr(joint_limits[0], '__len__'):
                    # Format: [[lower_limits], [upper_limits]]
                    lower_limit = float(joint_limits[0][0]) if len(joint_limits[0]) > 0 else -np.pi
                    upper_limit = float(joint_limits[1][0]) if len(joint_limits[1]) > 0 else np.pi
                else:
                    # Format: [lower_limit, upper_limit] or other format
                    lower_limit = float(joint_limits[0]) if len(joint_limits) > 0 else -np.pi
                    upper_limit = float(joint_limits[1]) if len(joint_limits) > 1 else np.pi
            except:
                # Fallback for any format issues
                lower_limit = -np.pi
                upper_limit = np.pi
                print(f"     Limits: [UNKNOWN FORMAT: {joint_limits}]")
                continue

            print(f"     Limits: [{lower_limit:.3f}, {upper_limit:.3f}]")
            print(f"     Homestate: {target_angle:.3f}")

            # Check if homestate is within limits
            if target_angle < lower_limit or target_angle > upper_limit:
                homestate_issues.append((joint_name, target_angle, (lower_limit, upper_limit)))
                print(f"     ❌ HOMESTATE OUT OF LIMITS!")
            else:
                print(f"     ✅ Homestate within limits")
        
        if homestate_issues:
            print(f"\n❌ HOMESTATE ISSUES FOUND:")
            for joint_name, angle, limits in homestate_issues:
                print(f"   {joint_name}: {angle:.3f} not in [{limits[0]:.3f}, {limits[1]:.3f}]")
        else:
            print(f"\n✅ All homestate values are within joint limits")
        
        # Test moving to homestate
        print(f"\n🚀 TESTING HOMESTATE MOVEMENT:")
        try:
            robot.move_to_homestate()
            print(f"   ✅ move_to_homestate() executed successfully")
            
            # Step the simulation a few times to see if it stabilizes
            for i in range(10):
                scene.step()
            
            # Check current joint positions
            current_qpos = robot.left_entity.get_qpos()
            arm_qpos = current_qpos[:len(robot.left_arm_joints_name)]
            
            print(f"   Current arm positions: {[f'{q:.3f}' for q in arm_qpos]}")
            print(f"   Target homestate:     {[f'{h:.3f}' for h in robot.left_homestate]}")
            
            # Check if positions are close to homestate
            position_errors = np.abs(np.array(arm_qpos) - np.array(robot.left_homestate))
            max_error = np.max(position_errors)
            
            print(f"   Maximum position error: {max_error:.3f}")
            
            if max_error > 0.1:  # 0.1 radian threshold
                print(f"   ⚠️  Large position error - joints may not be reaching homestate")
            else:
                print(f"   ✅ Joints successfully reached homestate")
                
        except Exception as e:
            print(f"   ❌ Error during homestate movement: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Test forward kinematics at homestate
        print(f"\n🎯 TESTING FORWARD KINEMATICS:")
        try:
            # Get end effector pose at homestate
            ee_pose = robot.get_left_ee_pose()
            print(f"   End effector position: [{ee_pose.p[0]:.3f}, {ee_pose.p[1]:.3f}, {ee_pose.p[2]:.3f}]")
            print(f"   End effector quaternion: [{ee_pose.q[0]:.3f}, {ee_pose.q[1]:.3f}, {ee_pose.q[2]:.3f}, {ee_pose.q[3]:.3f}]")
            print(f"   ✅ Forward kinematics working")
            
        except Exception as e:
            print(f"   ❌ Forward kinematics error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Test a simple planning operation
        print(f"\n🗺️  TESTING SIMPLE PLANNING:")
        try:
            # Get current pose and try to plan to the same pose (should be trivial)
            current_pose = robot.get_left_ee_pose()
            target_pose = [current_pose.p[0], current_pose.p[1], current_pose.p[2] + 0.01,  # Slightly higher
                          current_pose.q[0], current_pose.q[1], current_pose.q[2], current_pose.q[3]]
            
            print(f"   Planning to target: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]")
            
            result = robot.left_plan_path(target_pose)
            
            if result["status"] == "Success":
                print(f"   ✅ Planning successful!")
                print(f"   Plan steps: {result['position'].shape[0] if 'position' in result else 'N/A'}")
            else:
                print(f"   ❌ Planning failed: {result['status']}")
                
        except Exception as e:
            print(f"   ❌ Planning error: {str(e)}")
            if "Eigenvalues did not converge" in str(e):
                print(f"   🎯 FOUND THE EIGENVALUE ERROR!")
                print(f"   💡 This suggests numerical instability in the planning algorithm")
            import traceback
            traceback.print_exc()
        
        # Summary and recommendations
        print(f"\n" + "=" * 60)
        print(f"📋 DIAGNOSTIC SUMMARY:")
        
        if homestate_issues:
            print(f"   ❌ HOMESTATE ISSUES: {len(homestate_issues)} joints out of limits")
            print(f"   💡 RECOMMENDATION: Fix homestate values to be within joint limits")
        
        print(f"   💡 POTENTIAL SOLUTIONS:")
        print(f"   1. Adjust homestate to a more stable configuration")
        print(f"   2. Check CuRobo configuration for numerical stability")
        print(f"   3. Verify joint limits in URDF are reasonable")
        print(f"   4. Consider using a different retract_config in CuRobo")
        
        return len(homestate_issues) == 0
        
    except Exception as e:
        print(f"❌ Failed to create robot: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        scene = None

if __name__ == "__main__":
    success = diagnose_eigenvalue_error()
    exit(0 if success else 1)
