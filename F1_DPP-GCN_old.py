#!/usr/bin/env python
from __future__ import print_function

import argparse
import glob
import inspect
import os
import pickle
import random
import resource
import shutil
import sys
import time
import traceback
import warnings
from collections import OrderedDict

import numpy as np
# torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from torchlight import DictAction
from tqdm import tqdm

# === 修改：导入 KLLoss 和 CrossEntropyLoss ===
# KLLoss 用于你原来的 L_proto
# CrossEntropyLoss 用于 L_ce 和 L_adv
from KLLoss import KLLoss
from model.ctrgcn import TextCLIP
from text.Text_Prompt import * #
from tools import * #

# === 修改：加载所有文本原型 ===
# 1. Branch A (主任务) 原型
classes, num_text_aug, text_dict = text_prompt_part_description_4part_clip()
# 2. Branch B (对抗任务) 原型
confounder_tokens = create_confounder_prototypes()

device = "cuda" if torch.cuda.is_available() else "cpu"

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, message=".*reduction: 'mean'.*")


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/DPP_GCN.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=0,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    
    # === 修改：替换旧的 loss_alpha ===
    # parser.add_argument('--loss-alpha', type=float, default=1.0) #
    parser.add_argument('--lambda-proto', type=float, default=1.0, help='Weight for L_proto (contrastive loss)')
    parser.add_argument('--lambda-adv', type=float, default=0.5, help='Weight for L_adv (adversarial loss)')

    return parser


class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()

        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        
        # === 修改：加载所有文本原型到 GPU ===
        self.load_text_prototypes()
        
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_data()
            self.load_optimizer()
        self.lr = self.arg.base_lr
        self.best_F1 = 0
        self.best_F1_epoch = 0
        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                # === 修改：DPP_GCN 本身是 DataParallel，text_dict 不再是模型 ===
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)


    # === 新增：加载所有文本原型 ===
    def load_text_prototypes(self):
        self.print_log("Loading text prototypes to GPU...")
        # 1. 加载 CLIP (仅用于编码)
        clip_model, _ = clip.load(self.arg.model_args['head'][0], device)
        clip_model.eval()

        # 2. 编码 Branch A 步态原型
        with torch.no_grad():
            E_proto_gait_tokens = classes.to(device) #
            # (10, 768)
            self.E_proto_gait = clip_model.encode_text(E_proto_gait_tokens).float()
            
            # (10, 768) -> (5, 2, 768) 以便索引
            # 5 = 1个全局 + 4个部位
            # 2 = 正常/抑郁
            self.E_proto_gait = self.E_proto_gait.view(num_text_aug, 2, -1)

        # 3. 编码 Branch B 混淆原型
        self.E_conf_all = {}
        confounder_keys = ['cloth', 'angle', 'height', 'direction']
        with torch.no_grad():
            for key in confounder_keys:
                tokens = confounder_tokens[key].to(device)
                self.E_conf_all[key] = clip_model.encode_text(tokens).float()
        
        del clip_model # 释放 CLIP 模型内存
        self.print_log("Text prototypes loaded and encoded.")


    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker, #
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker, #
            drop_last=False,
            worker_init_fn=init_seed)


    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        print(self.model)
        
        # === 修改：定义所有损失函数 ===
        # L_ce: 主分类损失 (保持不变)
        self.loss_ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).cuda(output_device))
        
        # L_proto: 主对比损失 (保持不变)
        self.loss_proto_kl = KLLoss().cuda(output_device)
        
        # L_adv: 新的对抗性损失 (使用标准交叉熵)
        self.loss_adv_ce = nn.CrossEntropyLoss()
        # =================================

        # === 修改：不再需要 model_text_dict ===
        # self.model_text_dict = nn.ModuleDict() ...
        # (相关代码已删除)
        # ===================================

        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)


    def load_optimizer(self):
        # === 修改：优化器只包含 self.model 的参数 ===
        params = self.model.parameters()

        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

        # (OneCycleLR 保持不变)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.arg.base_lr, # <-- 修改：使用 base_lr 作为 peak rate
            steps_per_epoch=len(self.data_loader['train']),
            epochs=self.arg.num_epoch,
            pct_start=self.arg.warm_up_epoch / self.arg.num_epoch # 根据 warmup epoch 设置
        )
        self.print_log(f'using OneCycleLR with peak rate of {self.arg.base_lr}')


    def save_arg(self):
        # ... (save_arg 函数保持不变) ...
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)


    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()


    def print_time(self):
        # ... (print_time 函数保持不变) ...
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        # ... (print_log 函数保持不变) ...
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        # ... (record_time 函数保持不变) ...
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        # ... (split_time 函数保持不变) ...
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']

        loss_value = []
        loss_ce_value = []
        loss_proto_value = []
        loss_adv_value = []
        f1_value = []
        precision_value = []
        recall_value = []

        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        # === 准备 Branch B 混淆原型 (从 self 中获取) ===
        conf_keys = ['cloth', 'angle', 'height', 'direction']
        E_conf = [self.E_conf_all[k] for k in conf_keys]

        for batch_idx, (data, labels_dict, index) in enumerate(process):
            self.global_step += 1
            
            # === 修改：加载所有数据和标签到 GPU ===
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                # 主标签 (抑郁/正常)
                label_dep = labels_dict['label_dep'].long().cuda(self.output_device)
                # 混淆标签
                label_cloth = labels_dict['label_cloth'].long().cuda(self.output_device)
                label_angle = labels_dict['label_angle'].long().cuda(self.output_device)
                label_height = labels_dict['label_height'].long().cuda(self.output_device)
                label_direction = labels_dict['label_direction'].long().cuda(self.output_device)
                
                # 打包混淆标签
                labels_conf = [label_angle, label_cloth, label_height, label_direction]
            
            timer['dataloader'] += self.split_time()

            # forward
            with torch.cuda.amp.autocast():
                # === 修改：DPP_GCN 的新输出 ===
                logits_ce, feature_dict, logit_scale, part_features, logits_adv = self.model(data)
                
                # --- 1. 计算 L_proto (主对比损失，逻辑同) ---
                label_g = gen_label(label_dep)
                ground_truth = torch.tensor(label_g, dtype=torch.float, device=device) # KLLoss 需要 float
                
                loss_te_list = []

                # === 修改：移除了全局特征的损失计算 (f_global_proto) ===
                # (旧的 F1_main_multipart_CTRGCN.py 脚本只循环了 4 次)
                # ======================================================

                # 4个部位特征 (来自 part_features)
                for ind in range(4): # 0, 1, 2, 3
                    f_part = part_features[ind] 
                    
                    # (索引 ind+1 对应 text_dict[1] 到 text_dict[4])
                    E_text_part = self.E_proto_gait[ind + 1, label_dep, :] 
                    
                    # (索引 ind+1 对应 logit_scale 的第 1 到 4 号索引)
                    logits_img_part, logits_text_part = create_logits(f_part, E_text_part, logit_scale[:, ind + 1].mean())
                    
                    loss_img_p = self.loss_proto_kl(logits_img_part, ground_truth)
                    loss_text_p = self.loss_proto_kl(logits_text_part, ground_truth)
                    loss_te_list.append((loss_img_p + loss_text_p) / 2)
                
                # (L_proto 现在只包含 4 个部分的损失)
                loss_proto = sum(loss_te_list) / len(loss_te_list) #

                # --- 2. 计算 L_ce (主分类损失，逻辑同) ---
                logits_ce = logits_ce.float()
                loss_ce = self.loss_ce(logits_ce, label_dep)
                
                # --- 3. 新增：计算 L_adv (对抗性损失) ---
                loss_adv_list = []
                # 遍历4个混淆因素
                for i in range(len(logits_adv)): 
                    logits_conf = logits_adv[i] # e.g., (B, 8) for angle
                    labels_conf_gt = labels_conf[i] # e.g., (B) for angle
                    loss_adv_list.append(self.loss_adv_ce(logits_conf, labels_conf_gt))
                
                loss_adv = sum(loss_adv_list) / len(loss_adv_list)

            # === 修改：合并所有损失 ===
            loss = loss_ce + self.arg.lambda_proto * loss_proto + self.arg.lambda_adv * loss_adv

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # 更新学习率 (OneCycleLR 在 batch 级别更新)
            self.scheduler.step()  # 更新OneCycleLR学习率

            loss_value.append(loss.data.item())
            loss_ce_value.append(loss_ce.data.item())
            loss_proto_value.append(loss_proto.data.item())
            loss_adv_value.append(loss_adv.data.item())
            
            timer['model'] += self.split_time()

            # (F1, Precision, Recall 计算保持不变)
            predict_label = torch.argmax(logits_ce.data, 1)
            precision = precision_score(label_dep.cpu(), predict_label.cpu(), zero_division=0, average='binary')
            recall = recall_score(label_dep.cpu(), predict_label.cpu(), zero_division=0, average='binary')
            f1 = f1_score(label_dep.cpu(), predict_label.cpu(), zero_division=0, average='binary')

            self.train_writer.add_scalar('f1', f1, self.global_step)
            self.train_writer.add_scalar('precision', precision, self.global_step)
            self.train_writer.add_scalar('recall', recall, self.global_step)
            
            f1_value.append(f1)
            precision_value.append(precision)
            recall_value.append(recall)

            self.train_writer.add_scalar('loss_total', loss.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_ce', loss_ce.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_proto', loss_proto.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_adv', loss_adv.data.item(), self.global_step)
            
            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # (日志记录保持不变，增加了新的损失项)
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f} (CE: {:.4f}, Proto: {:.4f}, Adv: {:.4f}).'.format(
                np.mean(loss_value), np.mean(loss_ce_value), 
                np.mean(loss_proto_value), np.mean(loss_adv_value)
            ))
        self.print_log(
            '\tMean training F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}.'.format(
                np.mean(f1_value), np.mean(precision_value), np.mean(recall_value)
            )
        )
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights,
                       self.arg.model_saved_name + '-' + str(epoch + 1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            process = tqdm(self.data_loader[ln], ncols=40)

            all_labels = []  # 用于存储所有的标签
            all_preds = []  # 用于存储所有的预测结果

            # === 修改：eval 时数据加载也需要匹配 feeder ===
            for batch_idx, (data, labels_dict, index) in enumerate(process):
                # ============================================
                
                # === 修改：从 labels_dict 中获取主标签 ===
                label = labels_dict['label_dep']
                # ========================================
                
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)

                    # === 修改：DPP_GCN 在 eval 模式下只使用第一个输出来计算 L_ce ===
                    output, _, _, _, _ = self.model(data)
                    loss = self.loss_ce(output, label)
                    # ==========================================================

                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    # (F1, Precision, Recall 计算保持不变)
                    predict_label = torch.argmax(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())

                    all_labels.extend(label.data.cpu().numpy())
                    all_preds.extend(predict_label.data.cpu().numpy())

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + ',' + str(true[i]) + '\n')
                            if x != true[i] and wrong_file is not None:
                                f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)

            # (F1, Precision, Recall 计算保持不变)
            precision = precision_score(all_labels, all_preds, zero_division=0, average='binary')
            recall = recall_score(all_labels, all_preds, zero_division=0, average='binary')
            f1 = f1_score(all_labels, all_preds, zero_division=0, average='binary')

            print('F1_score: ', f1, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
            self.print_log('\tMean {} Precision: {:.4f}, Recall: {:.4f}'.format(ln, precision, recall))

            if f1 > self.best_F1:
                self.best_F1 = f1
                self.best_F1_epoch = epoch + 1

            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log(
                '\tMean {} loss of {} batches: {}.'.format(ln, len(self.data_loader[ln]), np.mean(loss_value)))
            self.print_log('\tF1_Score: {:.2f}%'.format(f1 * 100))
            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

        # (关闭文件句柄)
        if wrong_file is not None:
            f_w.close()
        if result_file is not None:
            f_r.close()

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            
            # === 修改：OneCycleLR 不在 epoch 级别调整 ===
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch + 1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)

                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            # (test the best model 逻辑保持不变)
            print(self.best_F1_epoch)
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-' + str(self.best_F1_epoch) + '*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True

            # (日志记录保持不变)
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best F1: {self.best_F1}')
            self.print_log(f'Epoch number: {self.best_F1_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


if __name__ == '__main__':
    parser = get_parser()

    # load arg from config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.SafeLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()