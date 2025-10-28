import numpy as np
from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, sample_path, label_path=None, p_interval=1, split='train', random_choose=False,
                 random_shift=False, random_move=False, random_rot=False, window_size=-1, normalization=False,
                 debug=False, use_mmap=False, bone=False, vel=False):

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        # Add sample_path
        self.sample_path = sample_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()

        if normalization:
            self.get_mean_map()


    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)

        if self.split == 'train':
            self.data = npz_data['x_train']
            # self.label 存储的是主任务标签（抑郁/正常）
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = np.loadtxt(self.sample_path, dtype=str)
            
            # === 修改：加载所有混淆标签 ===
            # 这些键必须与 seq_transformation.py 中保存的一致
            self.label_cloth = npz_data['y_conf_cloth']
            self.label_angle = npz_data['y_conf_angle']
            self.label_height = npz_data['y_conf_height']
            self.label_direction = npz_data['y_conf_direction']

        elif self.split == 'test':
            self.data = npz_data['x_test']
            # self.label 存储的是主任务标签（抑郁/正常）
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = np.loadtxt(self.sample_path, dtype=str)
            
            # === 修改：加载所有混淆标签 ===
            # 这些键必须与 seq_transformation.py 中保存的一致
            self.label_cloth = npz_data['y_conf_cloth']
            self.label_angle = npz_data['y_conf_angle']
            self.label_height = npz_data['y_conf_height']
            self.label_direction = npz_data['y_conf_direction']

        else:
            raise NotImplementedError('data split only supports train/test')

        N, T, _ = self.data.shape
        # Change to 2D
        self.data = self.data.reshape((N, T, 1, 17, 2)).transpose(0, 4, 1, 3, 2)


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))


    def __len__(self):
        return len(self.label)


    def __iter__(self):
        return self


    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index] # 主标签（抑郁/正常）
        data_numpy = np.array(data_numpy)

        # === 修改：获取当前索引的所有混淆标签 ===
        label_cloth = self.label_cloth[index]
        label_angle = self.label_angle[index]
        label_height = self.label_height[index]
        label_direction = self.label_direction[index]

        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        if self.bone:
            from .bone_pairs import dgait_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in dgait_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        # === 修改：将所有标签打包成一个字典 ===
        labels_dict = {
            'label_dep': label,
            'label_cloth': label_cloth,
            'label_angle': label_angle,
            'label_height': label_height,
            'label_direction': label_direction
        }
        
        return data_numpy, labels_dict, index


    def top_k(self, score, top_k):
        # 此函数保持不变，因为它只评估主标签（self.label）
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    
    
    def get_labels(self):
        # 此函数保持不变，它返回主标签
        return self.label


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod