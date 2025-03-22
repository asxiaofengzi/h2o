import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from phc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
    SMPL_BONE_ORDER_NAMES, 
)
import joblib
from phc.utils.rotation_conversions import axis_angle_to_matrix
from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch
from torch.autograd import Variable
from tqdm import tqdm
import argparse

def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not 'mocap_framerate' in  entry_data:
        return 
    framerate = entry_data['mocap_framerate']


    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans, 
        "betas": betas,
        "fps": framerate
    }
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_root", type=str, default="/home/js/xiaofengzi/h2o/data/AMASS/AMASS_Complete")
    args = parser.parse_args()
    
    device = torch.device("cpu")

    # h1_rotation_axis = torch.tensor([[
    #     [0, 0, 1], # l_hip_yaw
    #     [1, 0, 0], # l_hip_roll
    #     [0, 1, 0], # l_hip_pitch
        
    #     [0, 1, 0], # kneel
    #     [0, 1, 0], # ankle
        
    #     [0, 0, 1], # r_hip_yaw
    #     [1, 0, 0], # r_hip_roll
    #     [0, 1, 0], # r_hip_pitch
        
    #     [0, 1, 0], # kneel
    #     [0, 1, 0], # ankle
        
    #     [0, 0, 1], # torso
        
    #     [0, 1, 0], # l_shoulder_pitch
    #     [1, 0, 0], # l_roll_pitch
    #     [0, 0, 1], # l_yaw_pitch
        
    #     [0, 1, 0], # l_elbow
        
    #     [0, 1, 0], # r_shoulder_pitch
    #     [1, 0, 0], # r_roll_pitch
    #     [0, 0, 1], # r_yaw_pitch
        
    #     [0, 1, 0], # r_elbow
    # ]]).to(device)
    g1_rotation_axis = torch.tensor([[
        [0, 0, 1], # l_hip_yaw
        [1, 0, 0], # l_hip_roll
        [0, 1, 0], # l_hip_pitch
        
        [0, 1, 0], # l_knee
        [0, 1, 0], # l_ankle_pitch
        [1, 0, 0], # l_ankle_roll
        
        [0, 0, 1], # r_hip_yaw
        [1, 0, 0], # r_hip_roll
        [0, 1, 0], # r_hip_pitch
        
        [0, 1, 0], # r_knee
        [0, 1, 0], # r_ankle_pitch
        [1, 0, 0], # r_ankle_roll
        
        [0, 0, 1], # waist_yaw
        
        [0, 1, 0], # l_shoulder_pitch
        [1, 0, 0], # l_shoulder_roll
        [0, 0, 1], # l_shoulder_yaw
        
        [0, 1, 0], # l_elbow
        [1, 0, 0], # l_wrist_roll
        
        [0, 1, 0], # r_shoulder_pitch
        [1, 0, 0], # r_shoulder_roll
        [0, 0, 1], # r_shoulder_yaw
        
        [0, 1, 0], # r_elbow
        [1, 0, 0], # r_wrist_roll
    ]]).to(device)

    # h1_joint_names = [ 'pelvis', 
    #                 'left_hip_yaw_link', 'left_hip_roll_link','left_hip_pitch_link', 'left_knee_link', 'left_ankle_link',
    #                 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link',
    #                 'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 
    #                 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link']

    # h1_joint_names_augment = h1_joint_names + ["left_hand_link", "right_hand_link"]
    # h1_joint_pick = ['pelvis', "left_knee_link", "left_ankle_link",  'right_knee_link', 'right_ankle_link', "left_shoulder_roll_link", "left_elbow_link", "left_hand_link", "right_shoulder_roll_link", "right_elbow_link", "right_hand_link",]
    # smpl_joint_pick = ["Pelvis",  "L_Knee", "L_Ankle",  "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand"]
    # h1_joint_pick_idx = [ h1_joint_names_augment.index(j) for j in h1_joint_pick]
    # smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    # 定义G1关节名称
    # g1_joint_names = [
    #     'pelvis', 
    #     'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
    #     'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link',
    #     'waist_yaw_joint',
    #     'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_link',
    #     'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link'
    # ]
    g1_joint_names = [
        'pelvis', 
        'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
        'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link',
        'torso_link',  # 腰部的body名称
        'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_rubber_hand',  # 注意这里是rubber_hand而不是link
        'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_rubber_hand'  # 注意这里是rubber_hand而不是link
    ]

    # 定义关节间对应关系
    g1_joint_names_augment = g1_joint_names + ["left_hand_link", "right_hand_link"]
    g1_joint_pick = ['pelvis', "left_knee_link", "left_ankle_pitch_link", 'right_knee_link', 'right_ankle_pitch_link', 
                     "left_shoulder_roll_link", "left_elbow_link", "left_hand_link", 
                     "right_shoulder_roll_link", "right_elbow_link", "right_hand_link"]
    smpl_joint_pick = ["Pelvis", "L_Knee", "L_Ankle", "R_Knee", "R_Ankle", 
                      "L_Shoulder", "L_Elbow", "L_Hand", 
                      "R_Shoulder", "R_Elbow", "R_Hand"]
    g1_joint_pick_idx = [g1_joint_names_augment.index(j) for j in g1_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]


    # 加载SMPL模型
    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
    smpl_parser_n.to(device)

    # 加载之前拟合好的形状参数
    shape_new, scale = joblib.load("data/g1/shape_optimized_v1.pkl")
    shape_new = shape_new.to(device)

    # 处理AMASS数据集
    amass_root = args.amass_root
    all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {"0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path for data_path in all_pkls}
    
    if len(key_name_to_pkls) == 0:
        raise ValueError(f"No motion files found in {amass_root}")

    # h1_fk = Humanoid_Batch(device = device)
    # 初始化G1正向运动学
    g1_fk = Humanoid_Batch(mjcf_file="resources/robots/g1/g1_23dof.xml", device=device)
    data_dump = {}
    pbar = tqdm(key_name_to_pkls.keys())
    # 遍历AMASS数据集
    for data_key in pbar:
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        if amass_data is None:
            print(f"No motion files found in {key_name_to_pkls[data_key]}")
            continue
        skip = int(amass_data['fps']//30)
        trans = torch.from_numpy(amass_data['trans'][::skip]).float().to(device)
        N = trans.shape[0]
        pose_aa_walk = torch.from_numpy(np.concatenate((amass_data['pose_aa'][::skip, :66], np.zeros((N, 6))), axis = -1)).float().to(device)

        # 获取SMPL关节位置和顶点
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, torch.zeros((1, 10)).to(device), trans)
        offset = joints[:, 0] - trans
        root_trans_offset = trans + offset

        # 初始化G1关节位置
        # pose_aa_h1 = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], 22, axis = 2), N, axis = 1)
        # pose_aa_h1[..., 0, :] = (sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()
        # pose_aa_h1 = torch.from_numpy(pose_aa_h1).float().to(device)
        # gt_root_rot = torch.from_numpy((sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()).float().to(device)

        pose_aa_g1 = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], 26, axis=2), N, axis=1)  # 23+3
        pose_aa_g1[..., 0, :] = (sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()
        pose_aa_g1 = torch.from_numpy(pose_aa_g1).float().to(device)
        gt_root_rot = torch.from_numpy((sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()).float().to(device)

        # dof_pos = torch.zeros((1, N, 19, 1)).to(device)

        # dof_pos_new = Variable(dof_pos, requires_grad=True)
        # optimizer_pose = torch.optim.Adadelta([dof_pos_new],lr=100)
         # 初始化关节角度
        dof_pos = torch.zeros((1, N, 23, 1)).to(device)  # G1有23个DOF
        dof_pos_new = Variable(dof_pos, requires_grad=True)
        optimizer_pose = torch.optim.Adadelta([dof_pos_new], lr=100)

        # 迭代优化关节角度
        for iteration in range(500):
            # verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            # pose_aa_h1_new = torch.cat([gt_root_rot[None, :, None], h1_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2).to(device)
            # fk_return = h1_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ])
            
            # diff = fk_return['global_translation_extend'][:, :, h1_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            # loss_g = diff.norm(dim = -1).mean() 
            # loss = loss_g
            verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            pose_aa_g1_new = torch.cat([gt_root_rot[None, :, None], g1_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis=2).to(device)
            fk_return = g1_fk.fk_batch(pose_aa_g1_new, root_trans_offset[None, ])
            
            diff = fk_return['global_translation_extend'][:, :, g1_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            loss_g = diff.norm(dim=-1).mean() 
            loss = loss_g
            
            
            pbar.set_description_str(f"{iteration} {loss.item() * 1000}")

            optimizer_pose.zero_grad()
            loss.backward()
            optimizer_pose.step()
            
            # dof_pos_new.data.clamp_(h1_fk.joints_range[:, 0, None], h1_fk.joints_range[:, 1, None])
            # 限制关节角度在可行范围内
            dof_pos_new.data.clamp_(g1_fk.joints_range[:, 0, None], g1_fk.joints_range[:, 1, None])
            

        # dof_pos_new.data.clamp_(h1_fk.joints_range[:, 0, None], h1_fk.joints_range[:, 1, None])
        # pose_aa_h1_new = torch.cat([gt_root_rot[None, :, None], h1_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2)
        # fk_return = h1_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ])
        # 最终处理和数据保存
        dof_pos_new.data.clamp_(g1_fk.joints_range[:, 0, None], g1_fk.joints_range[:, 1, None])
        pose_aa_g1_new = torch.cat([gt_root_rot[None, :, None], g1_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis=2)
        fk_return = g1_fk.fk_batch(pose_aa_g1_new, root_trans_offset[None, ])

        # 调整根关节高度
        root_trans_offset_dump = root_trans_offset.clone()
        root_trans_offset_dump[..., 2] -= fk_return.global_translation[..., 2].min().item() - 0.08
        
        # 保存数据
        data_dump[data_key]={
                "root_trans_offset": root_trans_offset_dump.squeeze().cpu().detach().numpy(),
                "pose_aa": pose_aa_g1_new.squeeze().cpu().detach().numpy(),   
                "dof": dof_pos_new.squeeze().detach().cpu().numpy(), 
                "root_rot": sRot.from_rotvec(gt_root_rot.cpu().numpy()).as_quat(),
                "fps": 30
                }
        
         # 测试一个样本，然后如需处理所有数据，请移除此行
        print(f"dumping {data_key} for testing, remove the line if you want to process all data")
        import ipdb; ipdb.set_trace()
        # joblib.dump(data_dump, "data/h1/test.pkl")
        joblib.dump(data_dump, "data/g1/test.pkl")
    
    # 保存所有处理过的动作数据
    import ipdb; ipdb.set_trace()
    # joblib.dump(data_dump, "data/h1/amass_all.pkl")
    joblib.dump(data_dump, "/home/js/xiaofengzi/h2o/data/g1/amass_all_torso.pkl")