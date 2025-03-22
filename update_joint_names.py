import re
import os
import sys

def update_joint_names(file_path, new_joint_names):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 使用正则表达式查找和替换g1_joint_names定义
    pattern = r"g1_joint_names\s*=\s*\[(.*?)\]"
    replacement = "g1_joint_names = [\n    " + ",\n    ".join([f"'{name}'" for name in new_joint_names]) + "\n]"
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # 写回文件
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"已更新 {file_path}")

# 统一的关节名称列表
unified_joint_names = [
    'floating_base_link',
    'left_hip_pitch_link',
    'left_hip_roll_link',
    'left_hip_yaw_link',
    'left_knee_link',
    'left_ankle_pitch_link',
    'left_ankle_roll_link',
    'right_hip_pitch_link',
    'right_hip_roll_link',
    'right_hip_yaw_link',
    'right_knee_link',
    'right_ankle_pitch_link',
    'right_ankle_roll_link',
    'waist_yaw_link',
    'left_shoulder_pitch_link',
    'left_shoulder_roll_link',
    'left_shoulder_yaw_link',
    'left_elbow_link',
    'left_wrist_roll_link',
    'right_shoulder_pitch_link',
    'right_shoulder_roll_link',
    'right_shoulder_yaw_link',
    'right_elbow_link',
    'right_wrist_roll_link',
]

# 更新两个文件
update_joint_names("scripts/data_process/grad_fit_g1.py", unified_joint_names)
update_joint_names("scripts/data_process/grad_fit_g1_shape.py", unified_joint_names)

print("完成! 两个文件中的g1_joint_names已统一")
