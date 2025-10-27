import os
import os.path as osp
import numpy as np
import pickle
import logging

root_path = './'
raw_data_file = osp.join(root_path, 'raw_data_test', 'raw_skes_data.pkl')
save_path = osp.join(root_path, 'denoised_data_test')

if not osp.exists(save_path):
    os.mkdir(save_path)

actors_info_dir = osp.join(save_path, 'actors_info')
if not osp.exists(actors_info_dir):
    os.mkdir(actors_info_dir)

missing_count = 0
# labels = []  # <-- 我们将用多个列表替换这个

# Set up loggers for various denoising processes
noise_len_logger = logging.getLogger('noise_length')
noise_len_logger.setLevel(logging.INFO)
noise_len_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_length.log')))

noise_spr_logger = logging.getLogger('noise_spread')
noise_spr_logger.setLevel(logging.INFO)
noise_spr_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_spread.log')))

noise_mot_logger = logging.getLogger('noise_motion')
noise_mot_logger.setLevel(logging.INFO)
noise_mot_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_motion.log')))

missing_skes_logger = logging.getLogger('missing_frames')
missing_skes_logger.setLevel(logging.INFO)
missing_skes_logger.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes.log')))


def get_label_from_name(ske_name):
    """
    Extracts label based on the first letter of ske_name.
    'D' -> label 2, 'N' -> label 1
    """
    # Depression
    if ske_name.startswith('D'):
        return 2
    # Normal
    elif ske_name.startswith('N'):
        return 1
    else:
        raise ValueError(f"Unknown ske_name format: {ske_name}")


# ==============================================================================
# === 新增：标签解析函数 ===
# ==============================================================================
def get_all_labels_from_name(ske_name):
    """
    从 ske_name (例如 'N_001_bg-cl_E2-back') 中解析所有标签。
    返回: (label_dep, label_cloth, label_angle, label_height, label_direction)
    """
    
    # 定义标签到整数的映射
    cloth_map = {'nm': 0, 'cl': 1, 'bg-cl': 2}
    angle_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
    height_map = {'1': 0, '2': 1}
    direction_map = {'front': 0, 'back': 1}
    
    try:
        # ske_name 格式: N_001_bg-cl_E2-back
        parts = ske_name.split('_')

        # 确保 parts 至少有4个元素
        if len(parts) < 4:
            raise ValueError(f"文件名格式不正确，期望至少4个部分，但只得到 {len(parts)}: {ske_name}")

        id_part = parts[0]        # 'N'
        cloth_part = parts[2]     # 'bg-cl'
        view_part = parts[3]      # 'E2-back'
        
        # 1. 抑郁/正常标签
        label_dep = get_label_from_name(id_part)
        
        # 2. 服装标签
        label_cloth = cloth_map[cloth_part]
        
        # 3. 解析视角部分 (例如 'E2-back')
        view_parts = view_part.split('-')
        if len(view_parts) < 2:
            raise ValueError(f"视角部分格式不正确: {view_part}")
        
        angle_height_part = view_parts[0] # 'E2'
        direction_part = view_parts[1]    # 'back'

        if len(angle_height_part) < 2:
            raise ValueError(f"角度高度部分格式不正确: {angle_height_part}")
        
        # 4. 角度标签
        label_angle = angle_map[angle_height_part[0]] # 'E'
        
        # 5. 高度标签
        label_height = height_map[angle_height_part[1]] # '2'
        
        # 6. 方向标签
        label_direction = direction_map[direction_part]
        
        return label_dep, label_cloth, label_angle, label_height, label_direction
        
    except Exception as e:
        print(f"解析 ske_name 失败: {ske_name}")
        print(f"错误: {e}")
        # 返回 None 或抛出异常，这里我们返回 None 并跳过
        return None, None, None, None, None


def get_one_actor_points(body_data, num_frames):
    """ Get joints for one actor (no color data needed). """
    joints = np.zeros((num_frames, 34), dtype=np.float32)  # 17 joints with x, y coordinates

    start, end = body_data['interval'][0], body_data['interval'][-1]
    joints[start:end + 1] = body_data['joints'].reshape(-1, 34)  # Only x, y

    return joints


def get_raw_denoised_data():
    """ Get denoised joint positions (no color data) from raw skeleton sequences and generate labels. """
    with open(raw_data_file, 'rb') as fr:  # load raw skeletons data
        raw_skes_data = pickle.load(fr)

    num_skes = len(raw_skes_data)
    print(f'Found {num_skes} available skeleton sequences.')

    raw_denoised_joints = []
    frames_cnt = []

    skes_names = []  # List to store ske_name
    
    # === 修改：为所有标签创建列表 ===
    labels_dep = []       # 抑郁/正常
    labels_cloth = []     # 服装
    labels_angle = []     # 角度
    labels_height = []    # 高度
    labels_direction = [] # 方向

    for idx, bodies_data in enumerate(raw_skes_data):
        ske_name = bodies_data['name']
        print(f'Processing {ske_name}')

        # === 修改：调用新的标签解析函数 ===
        l_dep, l_cloth, l_angle, l_h, l_dir = get_all_labels_from_name(ske_name)
        
        if l_dep is None: # 如果解析失败，跳过这个样本
            print(f"Skipping {ske_name} due to parsing error.")
            continue
            
        # === 修改：将标签添加到各自的列表 ===
        labels_dep.append(l_dep)
        labels_cloth.append(l_cloth)
        labels_angle.append(l_angle)
        labels_height.append(l_h)
        labels_direction.append(l_dir)

        # only 1 actor
        num_frames = bodies_data['num_frames']
        body_data = list(bodies_data['data'].values())[0]
        joints = get_one_actor_points(body_data, num_frames)

        raw_denoised_joints.append(joints)
        frames_cnt.append(num_frames)

        skes_names.append(ske_name)  # Append ske_name to the list

        if (idx + 1) % 1000 == 0:
            print(f'Processed: {100.0 * (idx + 1) / num_skes:.2f}% ({idx + 1} / {num_skes}), '
                  f'Missing count: {missing_count}')

    # Save the denoised skeleton data
    raw_skes_joints_pkl = osp.join(save_path, 'raw_denoised_joints.pkl')
    with open(raw_skes_joints_pkl, 'wb') as f:
        pickle.dump(raw_denoised_joints, f, pickle.HIGHEST_PROTOCOL)

    # Save frame counts
    frames_cnt = np.array(frames_cnt, dtype=int)
    np.savetxt(osp.join(save_path, 'frames_cnt.txt'), frames_cnt, fmt='%d')

    # === 修改：分别保存所有标签文件 ===
    np.savetxt(osp.join(save_path, 'label_dep.txt'), np.array(labels_dep, dtype=int), fmt='%d')
    print(f'Depression labels saved to {osp.join(save_path, "label_dep.txt")}')
    
    np.savetxt(osp.join(save_path, 'label_cloth.txt'), np.array(labels_cloth, dtype=int), fmt='%d')
    print(f'Cloth labels saved to {osp.join(save_path, "label_cloth.txt")}')
    
    np.savetxt(osp.join(save_path, 'label_angle.txt'), np.array(labels_angle, dtype=int), fmt='%d')
    print(f'Angle labels saved to {osp.join(save_path, "label_angle.txt")}')
    
    np.savetxt(osp.join(save_path, 'label_height.txt'), np.array(labels_height, dtype=int), fmt='%d')
    print(f'Height labels saved to {osp.join(save_path, "label_height.txt")}')
    
    np.savetxt(osp.join(save_path, 'label_direction.txt'), np.array(labels_direction, dtype=int), fmt='%d')
    print(f'Direction labels saved to {osp.join(save_path, "label_direction.txt")}')
    
    # -----------------------------------

    dname_file = osp.join(save_path, 'dname.txt')
    with open(dname_file, 'w') as f:
        for ske_name in skes_names:
            f.write(ske_name + '\n')  # Write each ske_name followed by a newline

    print(f'Saved raw denoised positions of {np.sum(frames_cnt)} frames into {raw_skes_joints_pkl}')
    print(f'Found {missing_count} files that have missing data')


if __name__ == '__main__':
    get_raw_denoised_data()