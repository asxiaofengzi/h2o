import os
from isaacgym import gymapi
from isaacgym import gymtorch
import sys
import numpy as np
import torch
import joblib
import argparse
import time

import math

sys.path.append(os.getcwd())

def visualize_with_isaac_gym(model_path, motion_data, motion_key=None, fps=30):
    """使用Isaac Gym可视化G1机器人动作"""
    
    # 初始化gym
    gym = gymapi.acquire_gym()
    
    # 创建仿真
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / fps
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    
    # 添加地面平面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.distance = 0
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0.0
    
    # 创建仿真器
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("创建仿真失败")
        return
    
    # 添加地面
    gym.add_ground(sim, plane_params)
    
    # 选择动作
    if motion_key is None:
        motion_key = list(motion_data.keys())[0]
    print(f"可视化动作: {motion_key}")
    
    motion = motion_data[motion_key]
    pose_aa = motion['pose_aa']
    root_trans = motion['root_trans_offset']
    dof = motion.get('dof', None)
    motion_fps = motion.get('fps', 30)
    
    # 加载资源
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.angular_damping = 0.0
    asset_options.max_angular_velocity = 1000.0
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    
    g1_asset = gym.load_asset(sim, os.path.dirname(model_path), os.path.basename(model_path), asset_options)
    
    # 创建环境
    env_spacing = 2.0
    envs_per_row = 1
    env_count = 1
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    
    # 创建环境
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    
    # 添加actor
    initial_pose = gymapi.Transform()
    initial_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)  # 起始位置
    initial_pose.r = gymapi.Quat(0, 0, 0, 1)     # 起始旋转
    
    g1_handle = gym.create_actor(env, g1_asset, initial_pose, "g1", 0, 1)
    
    # 获取DOF属性
    dof_props = gym.get_actor_dof_properties(env, g1_handle)
    # 设置为位置控制模式
    for i in range(len(dof_props)):
        dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
        dof_props['stiffness'][i] = 10000.0
        dof_props['damping'][i] = 200.0
    
    gym.set_actor_dof_properties(env, g1_handle, dof_props)
    
    # 获取DOF数量和名称
    num_dofs = gym.get_actor_dof_count(env, g1_handle)
    print(f"G1有 {num_dofs} 个DOF")
    
    dof_names = []
    for i in range(num_dofs):
        name = gym.get_asset_dof_name(g1_asset, i)
        dof_names.append(name)
    print(f"DOF名称: {dof_names}")
    
    # 创建查看器
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1280
    cam_props.height = 720
    viewer = gym.create_viewer(sim, cam_props)
    if viewer is None:
        print("创建查看器失败")
        gym.destroy_sim(sim)
        return
    
    # 设置相机位置
    cam_pos = gymapi.Vec3(3, 3, 2)
    cam_target = gymapi.Vec3(0, 0, 1)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    
    # 可视化循环
    n_frames = pose_aa.shape[0]
    print(f"准备播放 {n_frames} 帧的动作...")
    
    # 主循环
    for i in range(n_frames):
        # 中断检查
        if gym.query_viewer_has_closed(viewer):
            break
        
        # 设置root位置
        root_pos = root_trans[i]
        root_state = gym.get_actor_rigid_body_states(env, g1_handle, gymapi.STATE_POS)
        root_state['pose']['p'][0][0] = root_pos[0]
        root_state['pose']['p'][0][1] = root_pos[1]
        root_state['pose']['p'][0][2] = root_pos[2]
        
        # 设置root旋转（从轴角转换为四元数）
        rot = pose_aa[i, 0]
        norm = np.linalg.norm(rot)
        if norm > 0:
            axis = rot / norm
            angle = norm
            quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(axis[0], axis[1], axis[2]), angle)
            root_state['pose']['r'][0][0] = quat.w
            root_state['pose']['r'][0][1] = quat.x
            root_state['pose']['r'][0][2] = quat.y
            root_state['pose']['r'][0][3] = quat.z
        
        # 应用根部状态
        gym.set_actor_rigid_body_states(env, g1_handle, root_state, gymapi.STATE_POS)
        
        # 设置关节角度
        if dof is not None and i < len(dof):
            dof_positions = np.zeros(num_dofs)
            
            # 将DOF值从弧度转换为关节角度
            # 这里假设dof是shape为(n_frames, num_dofs)的numpy数组
            for j in range(min(num_dofs, dof.shape[1])):
                dof_positions[j] = dof[i, j]
            
            # 应用DOF位置
            gym.set_actor_dof_position_targets(env, g1_handle, dof_positions)
        
        # 步进仿真
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # 步进图形
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        
        # 等待
        gym.sync_frame_time(sim)
        
        # 显示进度
        if i % 50 == 0:
            print(f"播放帧 {i}/{n_frames}")
    
    # 清理
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

def main():
    parser = argparse.ArgumentParser(description="G1机器人动作可视化")
    parser.add_argument("--motion_file", type=str, default="/home/js/xiaofengzi/h2o/data/g1/amass_all.pkl", 
                        help="动作文件路径")
    parser.add_argument("--model_path", type=str, default="/home/js/xiaofengzi/h2o/legged_gym/resources/robots/g1/g1_23dof.xml", 
                        help="G1模型XML文件路径")
    parser.add_argument("--motion_key", type=str, default=None, 
                        help="要可视化的特定动作键（默认使用第一个）")
    parser.add_argument("--fps", type=int, default=30, 
                        help="动作帧率")
    args = parser.parse_args()
    
    # 加载动作数据
    print(f"加载动作文件: {args.motion_file}")
    motion_data = joblib.load(args.motion_file)
    print(f"找到 {len(motion_data)} 个动作")
    
    # 打印可用的动作键
    print("可用动作:")
    for i, key in enumerate(motion_data.keys()):
        print(f"{i}: {key}")
    
    # 开始可视化
    visualize_with_isaac_gym(args.model_path, motion_data, args.motion_key, args.fps)

if __name__ == "__main__":
    main()