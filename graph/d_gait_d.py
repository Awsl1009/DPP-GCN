import sys

sys.path.extend(['../'])
from graph import tools

num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7),
                    (6, 8), (8, 10), (7, 9), (9, 11), (6, 12), (7, 13),
                    (12, 13), (12, 14), (14, 16), (13, 15), (15, 17)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

        # 新增功能：类似 AdjMatrixGraph
        self.self_loops = [(i, i) for i in range(self.num_node)]
        self.A_binary = tools.get_adjacency_matrix(self.neighbor, self.num_node)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.neighbor + self.self_loops, self.num_node)
        self.A_normalized = tools.normalize_adjacency_matrix(self.A_binary)

    
    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
