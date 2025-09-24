#!/usr/bin/env python3

import sys
import os
sys.path.append(os.getcwd())

import sapien
import numpy as np
from envs.beat_block_hammer import beat_block_hammer
from envs.utils.create_actor import create_actor, create_box
from envs.utils.actor_utils import Actor

def debug_target_pose():
    print("🔍 调试target_pose问题...")
    
    # 创建简单的场景
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    
    scene = engine.create_scene()
    scene.set_timestep(1 / 240.0)
    
    # 创建锤子
    print("\n1. 创建锤子...")
    hammer = create_actor(
        scene=scene,
        pose=sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105]),
        modelname="020_hammer",
        convex=True,
        model_id=0,
    )
    print(f"   锤子创建成功: {hammer}")
    print(f"   锤子配置: {type(hammer.config)}")
    
    # 检查contact_points
    print("\n2. 检查锤子的contact_points...")
    try:
        contact_points_pose = hammer.config.get("contact_points_pose", [])
        print(f"   contact_points_pose数量: {len(contact_points_pose)}")
        
        for i, cp in enumerate(contact_points_pose):
            print(f"   contact_point[{i}]: {cp}")
            
        # 测试get_contact_point方法
        print("\n3. 测试get_contact_point方法...")
        for i in range(len(contact_points_pose)):
            try:
                cp_matrix = hammer.get_contact_point(i, "matrix")
                cp_list = hammer.get_contact_point(i, "list")
                cp_pose = hammer.get_contact_point(i, "pose")
                print(f"   contact_point[{i}] matrix: {cp_matrix is not None}")
                print(f"   contact_point[{i}] list: {cp_list}")
                print(f"   contact_point[{i}] pose: {cp_pose}")
            except Exception as e:
                print(f"   contact_point[{i}] 错误: {e}")
                
        # 测试iter_contact_points方法
        print("\n4. 测试iter_contact_points方法...")
        try:
            for i, cp in hammer.iter_contact_points("list"):
                print(f"   iter contact_point[{i}]: {cp}")
        except Exception as e:
            print(f"   iter_contact_points 错误: {e}")
            
    except Exception as e:
        print(f"   检查contact_points时出错: {e}")
    
    # 创建方块
    print("\n5. 创建方块...")
    block = create_box(
        scene=scene,
        pose=sapien.Pose([0.1, 0.05, 0.76], [1, 0, 0, 0]),
        half_size=(0.025, 0.025, 0.025),
        color=(1, 0, 0),
        name="box",
        is_static=True,
    )
    print(f"   方块创建成功: {block}")
    
    # 检查functional_points
    print("\n6. 检查方块的functional_points...")
    try:
        functional_matrix = block.config.get("functional_matrix", [])
        print(f"   functional_matrix数量: {len(functional_matrix)}")
        
        for i, fp in enumerate(functional_matrix):
            print(f"   functional_point[{i}]: {fp}")
            
        # 测试get_functional_point方法
        print("\n7. 测试get_functional_point方法...")
        for i in range(len(functional_matrix)):
            try:
                fp_matrix = block.get_functional_point(i, "matrix")
                fp_list = block.get_functional_point(i, "list")
                fp_pose = block.get_functional_point(i, "pose")
                print(f"   functional_point[{i}] matrix: {fp_matrix is not None}")
                print(f"   functional_point[{i}] list: {fp_list}")
                print(f"   functional_point[{i}] pose: {fp_pose}")
            except Exception as e:
                print(f"   functional_point[{i}] 错误: {e}")
                
    except Exception as e:
        print(f"   检查functional_points时出错: {e}")
    
    print("\n✅ 调试完成!")

if __name__ == "__main__":
    debug_target_pose()
