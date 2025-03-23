"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

G1机器人动作可视化程序
"""
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import joblib
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from phc.utils.motion_lib_g1 import MotionLibG1  # 改为G1的运动库
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.flags import flags

flags.test = True
flags.im_eval = True

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

# 简单的资产描述符，用于从列表中选择
class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments

# 更改为G1的XML和URDF文件路径
g1_xml = "legged_gym/resources/robots/g1/g1_23dof.xml"
g1_urdf = "legged_gym/resources/robots/g1/g1_23dof.urdf"
asset_descriptors = [
    AssetDesc(g1_urdf, False),
]
sk_tree = SkeletonTree.from_mjcf(g1_xml)

# 更改为G1的运动文件路径
motion_file = "/home/js/xiaofengzi/h2o/data/g1/test.pkl"
if os.path.exists(motion_file):
    print(f"加载 {motion_file}")
else:
    raise ValueError(f"运动文件 {motion_file} 不存在！请先运行 grad_fit_g1.py。")

# 解析参数
args = gymutil.parse_arguments(description="关节动画：动画自由度范围",
                              custom_parameters=[{
                                  "name": "--asset_id",
                                  "type": int,
                                  "default": 0,
                                  "help": "资产ID (0 - %d)" % (len(asset_descriptors) - 1)
                              }, {
                                  "name": "--speed_scale",
                                  "type": float,
                                  "default": 1.0,
                                  "help": "动画速度缩放"
                              }, {
                                  "name": "--show_axis",
                                  "action": "store_true",
                                  "help": "可视化DOF轴"
                              }])

if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
    print("*** 指定的asset_id无效。有效范围是0到%d" % (len(asset_descriptors) - 1))
    quit()

# 初始化gym
gym = gymapi.acquire_gym()

# 配置模拟
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

if not args.use_gpu_pipeline:
    print("警告：强制使用CPU管线。")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** 创建模拟失败")
    quit()

# 添加地平面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# 创建查看器
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** 创建查看器失败")
    quit()

# 加载资产
asset_root = "./"
asset_file = asset_descriptors[args.asset_id].file_name

asset_options = gymapi.AssetOptions()
asset_options.use_mesh_materials = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS  # 设置为位置控制模式

print("从'%s'加载资产'%s'" % (asset_root, asset_file))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# 设置环境网格
num_envs = 1
num_per_row = 5
spacing = 5
env_lower = gymapi.Vec3(-spacing, spacing, 0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# 定位相机
cam_pos = gymapi.Vec3(0, -10.0, 3)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 缓存有用的句柄
envs = []
actor_handles = []

num_dofs = gym.get_asset_dof_count(asset)
print(f"G1有 {num_dofs} 个DOF")
print("创建%d个环境" % num_envs)
for i in range(num_envs):
    # 创建环境
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # 添加演员
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0, 0.0)
    pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

    actor_handle = gym.create_actor(env, asset, pose, "g1_actor", i, 1)
    actor_handles.append(actor_handle)
    
    # 获取DOF属性并设置为更高的控制刚度和阻尼
    dof_props = gym.get_actor_dof_properties(env, actor_handle)
    for j in range(len(dof_props)):
        dof_props['driveMode'][j] = gymapi.DOF_MODE_POS
        dof_props['stiffness'][j] = 10000.0
        dof_props['damping'][j] = 200.0
    
    gym.set_actor_dof_properties(env, actor_handle, dof_props)

    # 设置默认DOF位置
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

gym.prepare_sim(sim)

device = (torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu"))

# 创建G1的运动库
# 注意：你可能需要创建一个MotionLibG1类，类似于MotionLibH1
# 如果MotionLibG1不存在，你可以暂时使用MotionLibH1，但要确保它与G1的关节结构兼容
try:
    from phc.utils.motion_lib_g1 import MotionLibG1
    motion_lib = MotionLibG1(motion_file=motion_file, device=device, masterfoot_conifg=None, fix_height=False, multi_thread=False, mjcf_file=g1_xml)
except ImportError:
    # 如果没有专门为G1创建的运动库，可以临时使用H1的
    from phc.utils.motion_lib_h1 import MotionLibH1
    print("警告：使用H1的运动库代替G1的。这可能会导致关节映射问题。")
    motion_lib = MotionLibH1(motion_file=motion_file, device=device, masterfoot_conifg=None, fix_height=False, multi_thread=False, mjcf_file=g1_xml)

num_motions = 1
curr_start = 0
motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False)
motion_keys = motion_lib.curr_motion_keys

current_dof = 0
speeds = np.zeros(num_dofs)

time_step = 0
rigidbody_state = gym.acquire_rigid_body_state_tensor(sim)
rigidbody_state = gymtorch.wrap_tensor(rigidbody_state)
rigidbody_state = rigidbody_state.reshape(num_envs, -1, 13)

actor_root_state = gym.acquire_actor_root_state_tensor(sim)
actor_root_state = gymtorch.wrap_tensor(actor_root_state)

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "previous")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "next")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_G, "add")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "print")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_T, "next_batch")
motion_id = 0
motion_acc = set()

env_ids = torch.arange(num_envs).int().to(args.sim_device)

# 创建球体演员（用于可视化关节）
radius = 0.1
color = gymapi.Vec3(1.0, 0.0, 0.0)
sphere_params = gymapi.AssetOptions()
sphere_asset = gym.create_sphere(sim, radius, sphere_params)

while not gym.query_viewer_has_closed(viewer):
    # 步进物理
    try:
        motion_len = motion_lib.get_motion_length(motion_id).item()
        motion_time = time_step % motion_len
        motion_res = motion_lib.get_motion_state(torch.tensor([motion_id]).to(args.compute_device_id), torch.tensor([motion_time]).to(args.compute_device_id))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                    motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                    motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        
        if args.show_axis:
            gym.clear_lines(viewer)
            
        gym.clear_lines(viewer)
        gym.refresh_rigid_body_state_tensor(sim)
        
        # 可视化关节位置
        for pos_joint in rb_pos[0, 1:]:  # idx 0 是骨盆
            sphere_geom2 = gymutil.WireframeSphereGeometry(0.1, 4, 4, None, color=(1, 0.0, 0.0))
            sphere_pose = gymapi.Transform(gymapi.Vec3(pos_joint[0], pos_joint[1], pos_joint[2]), r=None)
            gymutil.draw_lines(sphere_geom2, gym, viewer, envs[0], sphere_pose) 
            
        # 更新根状态
        try:
            root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1).repeat(num_envs, 1)
            gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_states), gymtorch.unwrap_tensor(env_ids), len(env_ids))
        except Exception as e:
            print(f"设置根状态时出错: {e}")
            # 获取当前状态而不是使用gymtorch.unwrap_tensor
            root_state = gym.get_actor_rigid_body_states(envs[0], actor_handles[0], gymapi.STATE_POS)
            # 设置根位置
            root_state['pose']['p'][0][0] = root_pos[0, 0].item()
            root_state['pose']['p'][0][1] = root_pos[0, 1].item()
            root_state['pose']['p'][0][2] = root_pos[0, 2].item()
            # 设置根旋转
            root_state['pose']['r'][0][0] = root_rot[0, 0].item()  # w
            root_state['pose']['r'][0][1] = root_rot[0, 1].item()  # x
            root_state['pose']['r'][0][2] = root_rot[0, 2].item()  # y
            root_state['pose']['r'][0][3] = root_rot[0, 3].item()  # z
            # 应用根状态
            gym.set_actor_rigid_body_states(envs[0], actor_handles[0], root_state, gymapi.STATE_POS)

        gym.refresh_actor_root_state_tensor(sim)

        # 更新关节位置
        try:
            dof_state = torch.stack([dof_pos, torch.zeros_like(dof_pos)], dim=-1).squeeze().repeat(num_envs, 1)
            gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_state), gymtorch.unwrap_tensor(env_ids), len(env_ids))
        except Exception as e:
            print(f"设置DOF状态时出错: {e}")
            # 直接使用关节角度设置每个DOF
            for j in range(min(num_dofs, dof_pos.shape[1])):
                gym.set_dof_position_target(envs[0], actor_handles[0], j, dof_pos[0, j].item())

        gym.simulate(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.fetch_results(sim, True)
        
        # 更新查看器
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # 等待dt经过实时。
        # 这将物理模拟与渲染率同步。
        gym.sync_frame_time(sim)
        time_step += dt
        
    except Exception as e:
        print(f"主循环中出现错误: {e}")
        break

    # 处理键盘事件
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "previous" and evt.value > 0:
            motion_id = (motion_id - 1) % num_motions
            print(f"动作ID: {motion_id}. 动作长度: {motion_len:.3f}. 动作名称: {motion_keys[motion_id]}")
        elif evt.action == "next" and evt.value > 0:
            motion_id = (motion_id + 1) % num_motions
            print(f"动作ID: {motion_id}. 动作长度: {motion_len:.3f}. 动作名称: {motion_keys[motion_id]}")
        elif evt.action == "add" and evt.value > 0:
            motion_acc.add(motion_keys[motion_id])
            print(f"添加动作 {motion_keys[motion_id]}")
        elif evt.action == "print" and evt.value > 0:
            print(motion_acc)
        elif evt.action == "next_batch" and evt.value > 0:
            curr_start += num_motions
            motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
            motion_keys = motion_lib.curr_motion_keys
            print(f"下一批 {curr_start}")

        time_step = 0

print("完成")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)