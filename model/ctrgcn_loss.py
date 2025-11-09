import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import warnings

# 导入 GRL
from torch.autograd import Function

# 忽略所有警告
warnings.filterwarnings("ignore")


# === 新增：梯度反转层 (GRL) ===
class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # 存储 alpha 以备反向传播使用
        ctx.alpha = alpha
        return x.view_as(x)


    @staticmethod
    def backward(ctx, grad_output):
        # 反转梯度
        output = grad_output.neg() * ctx.alpha
        return output, None
    

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model


    def forward(self, text):
        return self.model.encode_text(text)
    

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(
        2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)


    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 2 or in_channels == 3 or in_channels == 9:  # change to 2, because the origin data is 2D-Pose.
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)


    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)


    def forward(self, x):
        x = self.bn(self.conv(x))
        return x
    

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
            
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)


    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations=[1, 2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)


    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


# === 新的 DPP-GCN 模型 ===
class DPP_GCN(nn.Module):
    """
    Disentangled Prior-Prompt GCN (DPP-GCN)
    集成了对抗性解耦 (DIFFER) 和分部位原型对齐 (PP-GCN) 的新模型。
    """
    def __init__(self, num_class=2, num_point=17, num_person=1, graph=None, graph_args=dict(), in_channels=2,
                 drop_out=0, adaptive=True, head=['ViT-L/14'],
                 # 新增参数:
                 num_confounders=4,      # 要对抗的混淆因素数量 (例如: 角度, 服装, 高度, 方向)
                 confounder_dims=None,   # 一个列表，包含每个混淆因素的类别数, e.g., [8, 3, 2, 2]
                 proto_dim=768,          # 原型文本的嵌入维度 (例如 ViT-L/14 为 768)
                 adv_head_dim=512,       # 对抗性分支的隐藏层维度
                 ):
        super(DPP_GCN, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A

        if confounder_dims is None:
            confounder_dims = [8, 3, 2, 2] # 默认为 [角度, 服装, 高度, 方向]

        self.num_class = num_class
        self.num_point = num_point
        self.head = head

        # --- 1. 骨架编码器 (Encoder) ---
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)

        encoder_out_dim = base_channel * 4

        # --- 2. 分支 A (主任务 - 完全同 Model_4part) ---

        # 2a. logit_scale (用于 1个全局 + 4个部位 对比损失)
        self.logit_scale = nn.Parameter(torch.ones(1, 5) * np.log(1 / 0.07))

        # 2b. 分部位原型头 (L_proto_part)
        self.part_list = nn.ModuleList([
            nn.Linear(encoder_out_dim, proto_dim) for _ in range(4)
        ])

        # 2c. 全局原型头 (L_proto_global)
        self.linear_head = nn.ModuleDict()
        for head_name in self.head:
            if head_name == 'ViT-L/14': # 假设 proto_dim 与 ViT-L/14 维度(768)一致
                self.linear_head[head_name] = nn.Linear(encoder_out_dim, proto_dim)
                conv_init(self.linear_head[head_name])

        # 2d. 分类头 (L_ce)
        self.fc = nn.Linear(encoder_out_dim, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        # --- 3. 分支 B (对抗任务) ---

        # 3a. 对抗性多任务头 (L_adv)
        # 这是一个多头分类器，用于预测所有混淆因素
        self.adv_head = nn.ModuleList()
        for num_c in confounder_dims:
            # 每一个混淆因素都有一个独立的分类头
            # z_global (256) -> 隐藏层 (adv_head_dim) -> 输出 (num_c)
            self.adv_head.append(
                nn.Sequential(
                    nn.Linear(encoder_out_dim, adv_head_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(adv_head_dim, num_c) # num_c 是该混淆因素的类别数
                )
            )

        # --- 4. 初始化 ---
        bn_init(self.data_bn, 1)
        for m in self.part_list:
                    weights_init(m)
        for head in self.adv_head:
            self.apply(weights_init)

    
    def forward(self, x, grl_alpha=1.0):
        # x 形状: (N, C, T, V, M)
        N, C, T, V, M = x.size()

        # --- 1. 编码器 (Encoder) ---
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)  # N是批次数量，C是坐标数，T是帧数，V是骨架数，M是人数
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        # GCN-TCN 堆栈
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)  # (N*M, 64, T/1, V)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)  # (N*M, 128, T/2, V)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # (N*M, 256, T/4, V)

        # (N*M, C_out, T_out, V) C_out=256, T_out=T/4
        c_out = x.size(1)

        # --- 2. 分支 A (主任务) ---

        # 2a. 分部位原型特征 (用于 L_proto)
        feature_part_in = x.view(N, M, c_out, T // 4, V)

        # 定义部位索引 (基于你原论文的 0-based 索引)
        head_list = torch.tensor([0, 1, 2, 3, 4]).long().to(x.device)
        hand_arm_list = torch.tensor([5, 6, 7, 8, 9, 10]).long().to(x.device) # 你的原索引是 5-11 (11是右手), 11也属于arm
        hip_list = torch.tensor([11, 12]).long().to(x.device) # 你的原索引 11, 12
        leg_foot_list = torch.tensor([13, 14, 15, 16]).long().to(x.device) # 你的原索引 13-16

        # (N, M, C_out, T_out, V_part) -> (N, M, C_out) -> (N*M, C_out)
        # 平均池化 (T_out, V_part)
        f_head = feature_part_in[:, :, :, :, head_list].mean(dim=[3, 4]).view(-1, c_out)
        f_arm = feature_part_in[:, :, :, :, hand_arm_list].mean(dim=[3, 4]).view(-1, c_out)
        f_hip = feature_part_in[:, :, :, :, hip_list].mean(dim=[3, 4]).view(-1, c_out)
        f_leg = feature_part_in[:, :, :, :, leg_foot_list].mean(dim=[3, 4]).view(-1, c_out)

        # (N*M, C_out=256) -> (N*M, proto_dim=768)
        proto_head = self.part_list[0](f_head)
        proto_arm = self.part_list[1](f_arm)
        proto_hip = self.part_list[2](f_hip)
        proto_leg = self.part_list[3](f_leg)

        # 打包分部位特征
        part_features = [proto_head, proto_arm, proto_hip, proto_leg]

        # 2b. 全局特征 (用于 L_ce 和 L_adv)
        # (N*M, C_out, T_out, V) -> (N*M, C_out)
        # 平均池化 (T_out, V)
        f_global = x.mean(dim=[2, 3]) # (N*M, 256)

        # 2c. 提取全局原型特征 (用于 L_proto_global)
        feature_dict = dict()
        for name in self.head:
            feature_dict[name] = self.linear_head[name](f_global)

        # 2d. 提取分类 logits (用于 L_ce)
        f_global_ce = self.drop_out(f_global)
        logits_ce = self.fc(f_global_ce)

        # --- 3. 分支 B (对抗任务) ---

        # 3a. GRL
        # alpha 在训练循环中传入，但这里为简化先设为 1.0
        # 我们只在训练时应用 GRL
        if self.training:
            f_global_adv = GradientReverseLayer.apply(f_global, grl_alpha) # <-- 2. 使用传入的 grl_alpha
        else:
            f_global_adv = f_global

        # 3b. 对抗性 logits (用于 L_adv)
        logits_adv = []
        for adv_classifier in self.adv_head:
            logits_adv.append(adv_classifier(f_global_adv))

        # --- 4. 返回所有输出 ---

        # logits_ce: [N*M, num_class] (用于 L_ce)
        # part_features: list[ [N*M, proto_dim], ... ] (4个部位, 用于 L_proto)
        # logits_adv: list[ [N*M, num_c1], [N*M, num_c2], ... ] (N个混淆因素, 用于 L_adv)
        return logits_ce, feature_dict, self.logit_scale, part_features, logits_adv