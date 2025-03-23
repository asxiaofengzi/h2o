import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from phc.utils import torch_utils
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser, 
)

import joblib
import torch
import torch.nn.functional as F
import math
from phc.utils.pytorch3d_transforms import axis_angle_to_matrix
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter1d
from tqdm.notebook import tqdm
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
# from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch, H1_ROTATION_AXIS
from phc.utils.torch_g1_humanoid_batch import Humanoid_Batch, G1_ROTATION_AXIS

# h1_joint_names = [ 'pelvis', 
#                    'left_hip_yaw_link', 'left_hip_roll_link','left_hip_pitch_link', 'left_knee_link', 'left_ankle_link',
#                    'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link',
#                    'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 
#                   'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link']
# G1关节名称定义
g1_joint_names = [
    'pelvis', 
    'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
    'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link',
    'torso_link',  # 腰部的body名称
    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_rubber_hand',  # 注意这里是rubber_hand而不是link
    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_rubber_hand'  # 注意这里是rubber_hand而不是link
]

# h1_fk = Humanoid_Batch(extend_head=True) # load forward kinematics model
# 加载G1正向运动学模型
g1_fk = Humanoid_Batch(mjcf_file="resources/robots/g1/g1_23dof.xml", extend_head=True) 

# #### Define corresonpdances between h1 and smpl joints
# h1_joint_names_augment = h1_joint_names + ["left_hand_link", "right_hand_link", "head_link"]
# h1_joint_pick = ['pelvis',  'left_hip_yaw_link', "left_knee_link", "left_ankle_link",  'right_hip_yaw_link', 'right_knee_link', 'right_ankle_link', "left_shoulder_roll_link", "left_elbow_link", "left_hand_link", "right_shoulder_roll_link", "right_elbow_link", "right_hand_link", "head_link"]
# smpl_joint_pick = ["Pelvis", "L_Hip",  "L_Knee", "L_Ankle",  "R_Hip", "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand", "Head"]
# h1_joint_pick_idx = [ h1_joint_names_augment.index(j) for j in h1_joint_pick]
# smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

# 定义G1和SMPL关节之间的对应关系
g1_joint_names_augment = g1_joint_names + ["left_hand_link", "right_hand_link", "head_link"]
g1_joint_pick = ["pelvis", 
                 "left_hip_pitch_link", 
                 "left_knee_link", 
                 "left_ankle_pitch_link", 
                 "right_hip_pitch_link", 
                 "right_knee_link", 
                 "right_ankle_pitch_link", 
                 "left_shoulder_roll_link", 
                 "left_elbow_link", 
                 "left_hand_link", 
                 "right_shoulder_roll_link", 
                 "right_elbow_link", 
                 "right_hand_link", 
                 "head_link"]
smpl_joint_pick = ["Pelvis", 
                   "L_Hip", 
                   "L_Knee", 
                   "L_Ankle", 
                   "R_Hip", 
                   "R_Knee", 
                   "R_Ankle", 
                  "L_Shoulder", 
                  "L_Elbow", 
                  "L_Hand", 
                  "R_Shoulder", 
                  "R_Elbow", 
                  "R_Hand", 
                  "Head"]
g1_joint_pick_idx = [g1_joint_names_augment.index(j) for j in g1_joint_pick]
smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]



#### 准备拟合变量
device = torch.device("cpu")
# pose_aa_h1 = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], 22, axis = 2), 1, axis = 1)
pose_aa_g1 = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], 26, axis=2), 1, axis=1)  # 23 + 3 (含hand和head)
# pose_aa_h1 = torch.from_numpy(pose_aa_h1).float()
pose_aa_g1 = torch.from_numpy(pose_aa_g1).float()

# dof_pos = torch.zeros((1, 19))
# pose_aa_h1 = torch.cat([torch.zeros((1, 1, 3)), H1_ROTATION_AXIS * dof_pos[..., None], torch.zeros((1, 2, 3))], axis = 1)
dof_pos = torch.zeros((1, 23))
pose_aa_g1 = torch.cat([torch.zeros((1, 1, 3)), G1_ROTATION_AXIS * dof_pos[..., None], torch.zeros((1, 2, 3))], axis=1)


root_trans = torch.zeros((1, 1, 3))    
# root_trans[0, 0, 2] = -0.1  # 向下调整初始高度，根据需要修改数值

###### prepare SMPL default pause for G1
pose_aa_stand = np.zeros((1, 72))
rotvec = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec()
pose_aa_stand[:, :3] = rotvec
pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Shoulder')] = sRot.from_euler("xyz", [0, 0, -np.pi/2],  degrees = False).as_rotvec()
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Shoulder')] = sRot.from_euler("xyz", [0, 0, np.pi/2],  degrees = False).as_rotvec()
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Elbow')] = sRot.from_euler("xyz", [0, -np.pi/2, 0],  degrees = False).as_rotvec()
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Elbow')] = sRot.from_euler("xyz", [0, np.pi/2, 0],  degrees = False).as_rotvec()
pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))

# 加载SMPL模型
smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")

###### 形状拟合过程
trans = torch.zeros([1, 3])
beta = torch.zeros([1, 10])
verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta , trans)
offset = joints[:, 0] - trans
root_trans_offset = trans + offset

# fk_return = h1_fk.fk_batch(pose_aa_h1[None, ], root_trans_offset[None, 0:1])
fk_return = g1_fk.fk_batch(pose_aa_g1[None, ], root_trans_offset[None, 0:1])


shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
scale = Variable(torch.ones([1]).to(device), requires_grad=True)
optimizer_shape = torch.optim.Adam([shape_new, scale],lr=0.1)


for iteration in range(1000):
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])
    root_pos = joints[:, 0]
    joints = (joints - joints[:, 0]) * scale + root_pos
    # diff = fk_return.global_translation_extend[:, :, h1_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
    # 在计算diff之前添加
    # print("g1_joint_names_augment:", g1_joint_names_augment)
    # print("g1_joint_pick:", g1_joint_pick)
    # print("g1_joint_pick_idx:", g1_joint_pick_idx)
    # print("fk_return.global_translation_extend.shape:", fk_return.global_translation_extend.shape)
    diff = fk_return.global_translation_extend[:, :, g1_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
    loss_g = diff.norm(dim = -1).mean() 
    loss = loss_g
    if iteration % 100 == 0:
        print(iteration, loss.item() * 1000)

    optimizer_shape.zero_grad()
    loss.backward()
    optimizer_shape.step()

# os.makedirs("data/h1", exist_ok=True)
# joblib.dump((shape_new.detach(), scale), "data/h1/shape_optimized_v1.pkl") # V2 has hip jointsrea
# print(f"shape fitted and saved to data/h1/shape_optimized_v1.pkl")
# 保存拟合结果
os.makedirs("data/g1", exist_ok=True)
joblib.dump((shape_new.detach(), scale), "data/g1/shape_optimized_v1.pkl")
print(f"shape fitted and saved to data/g1/shape_optimized_v1.pkl")