import os
import os.path as osp
import pickle

import numpy as np

root_path = './'
denoised_path = osp.join(root_path, 'denoised_data_test') # 确保这是你正确的路径
raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
frames_file = osp.join(denoised_path, 'frames_cnt.txt')

# === 修改：定义所有标签文件的路径 ===
label_dep_file = osp.join(denoised_path, 'label_dep.txt')
label_cloth_file = osp.join(denoised_path, 'label_cloth.txt')
label_angle_file = osp.join(denoised_path, 'label_angle.txt')
label_height_file = osp.join(denoised_path, 'label_height.txt')
label_direction_file = osp.join(denoised_path, 'label_direction.txt')
# --------------------------------

save_path = './processed_data_test' # 确保这是你正确的路径

if not osp.exists(save_path):
    os.mkdir(save_path)


def align_frames(skes_joints, frames_cnt):
    """ Align sequences with the same frame length """
    num_skes = len(skes_joints)
    max_num_frames = frames_cnt.max()  # Max frame length
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 34), dtype=np.float32)  # 34 bodies

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        padded_ske_joints = ske_joints
        aligned_skes_joints[idx, :num_frames] = padded_ske_joints  # Align the joint data

    return aligned_skes_joints


# === 修改：更新函数签名以接受所有标签 ===
def split_dataset(skes_joints, labels_dep, labels_cloth, labels_angle, 
                  labels_height, labels_direction, save_path):
    # Make sure the path exists
    if not osp.exists(save_path):
        os.makedirs(save_path)

    # 骨架数据
    test_x = skes_joints
    
    # 1. 主要标签 (抑郁/正常), 转换为 one-hot
    test_y_dep_onehot = one_hot_vector(labels_dep)

    # 2. 混淆标签 (保持为整数，用于对抗性训练)
    test_y_conf_cloth = labels_cloth
    test_y_conf_angle = labels_angle
    test_y_conf_height = labels_height
    test_y_conf_direction = labels_direction

    # Save the segmented dataset
    save_name = 'd_gait_test.npz'
    
    # === 修改：在 .npz 文件中保存所有标签 ===
    np.savez(osp.join(save_path, save_name), 
             x_test=test_x, 
             y_test=test_y_dep_onehot,         # 主要标签 (one-hot)
             y_dep_raw=labels_dep,               # 主要标签 (原始整数)
             y_conf_cloth=test_y_conf_cloth,    # 混淆标签: 服装
             y_conf_angle=test_y_conf_angle,    # 混淆标签: 角度
             y_conf_height=test_y_conf_height,  # 混淆标签: 高度
             y_conf_direction=test_y_conf_direction # 混淆标签: 方向
             )
    print(f"保存划分数据集到 {save_name}")
    print(f"  - x_test (骨架): {test_x.shape}")
    print(f"  - y_test (抑郁 one-hot): {test_y_dep_onehot.shape}")
    print(f"  - y_conf_cloth (服装): {test_y_conf_cloth.shape}")
    print(f"  - y_conf_angle (角度): {test_y_conf_angle.shape}")
    print(f"  - y_conf_height (高度): {test_y_conf_height.shape}")
    print(f"  - y_conf_direction (方向): {test_y_conf_direction.shape}")
# ---------------------------------------------


def one_hot_vector(labels):
    """ Convert labels to one-hot encoded vectors """
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 2))  # Assuming 2 classes (D, N)

    for idx, l in enumerate(labels):
        labels_vector[idx, l - 1] = 1  # Label should be 1 or 2, so adjust by subtracting 1

    # [1, 0] mean label 1 Normal
    # [0, 1] mean label 2 Depression
    return labels_vector


if __name__ == '__main__':
    # === 修改：加载所有标签文件 ===
    labels_dep = np.loadtxt(label_dep_file, dtype=int)
    labels_cloth = np.loadtxt(label_cloth_file, dtype=int)
    labels_angle = np.loadtxt(label_angle_file, dtype=int)
    labels_height = np.loadtxt(label_height_file, dtype=int)
    labels_direction = np.loadtxt(label_direction_file, dtype=int)
    # -----------------------------

    # Load frame counts and skeleton joints data
    frames_cnt = np.loadtxt(frames_file, dtype=int)
    with open(raw_skes_joints_pkl, 'rb') as fr:
        skes_joints = pickle.load(fr)  # Load denoised joints data

    # Perform sequence alignment
    skes_joints = align_frames(skes_joints, frames_cnt)

    # === 修改：将所有标签传递给 split_dataset ===
    split_dataset(skes_joints, labels_dep, labels_cloth, labels_angle, 
                  labels_height, labels_direction, save_path)
    # ---------------------------------------