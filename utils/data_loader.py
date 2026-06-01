import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from utils.utils import *
# from ptrans3 import PointTransformerV3, Point
# from ptrans1 import pointtransformer_enc_repro
from typing import Dict, List
from einops import rearrange
import random

from loss import isStateFree

def collate_fn_4ptrans(batch: List) -> Dict:
    """ General collate function used for dataloader.
    This collate function is used for point-transformer
    """
    batch_data = {key: [d[key] for d in batch] for key in batch[0]}
    
    for key in batch_data:
        if torch.is_tensor(batch_data[key][0]):
            batch_data[key] = torch.stack(batch_data[key])
    
    ## squeeze the first dimension of pos and feat
    offset, count = [], 0
    for item in batch_data['pos']:
        count += item.shape[0]
        offset.append(count)
    offset = torch.IntTensor(offset)
    batch_data['offset'] = offset
    
    return batch_data

class MomaTrajTrans():
    def __init__(self, norm_method: str, 
                 mu: torch.Tensor, 
                 std: torch.Tensor):
        self.normer = NORMER[norm_method]
        self.denormer = DENORMER[norm_method]
        self.mu = mu
        self.std = std
    
    def trans(self, wpsin):
        wps = wpsin.clone()
        wps, target = self.normer(wps)
        wps = (wps - self.mu.to(wps.device)) / self.std.to(wps.device)
        return wps, target
    
    def detrans(self, data):
        wps = data[0] * self.std.to(data[0].device) + self.mu.to(data[0].device)
        wps = self.denormer((wps, data[1]))
        return wps

class MomaDataset(Dataset):
    def __init__(self, filepath="data_train_big_raw.h5", 
                 norm_method="simple",
                 lib_path = "logs/trajlib32.npy",
                 mustd_path = None):
        
        self.filepath = filepath
        self.normer = NORMER[norm_method]
        self.denormer = DENORMER[norm_method]
        if lib_path is not None:
            self.trajlib = np.load(lib_path, allow_pickle=True)
            self.trajlib = torch.as_tensor(self.trajlib).float().contiguous()
        
        self.f = h5py.File(self.filepath, 'r')
        # self.wps = self.f["wps"]
        # self.occs = self.f["occs"]
        # self.sdf = self.f["sdf"]
        # self.min_idx = self.f["min_idx"]
        # self.boxes = self.f["boxes"]

        for k,v in self.f.items():
            setattr(self, k, v)
        
        self.length_ = len(self.f["wps"])
        self.wps_num = self.wps.shape[1]
        self.state_dim = self.wps.shape[2]
                    
        print("Dataset length:", self.length_)
        print("Occs shape:", self.occs.shape)
        print("SDF shape:", self.sdf.shape)
        print("Wps shape:", self.wps.shape)
        print("Min idx shape:", self.min_idx.shape)
        # print("Boxes shape:", self.boxes.shape)
        
        self.mu = None
        self.std = None
        self.transer = None
        if mustd_path is not None:
            mu_std = np.load(mustd_path, allow_pickle=True)
            mu_std = torch.as_tensor(mu_std).float().contiguous()
            self.mu = mu_std[0]
            self.std = mu_std[1]
        else:
            self.get_mustd_welford()
            self.transer = MomaTrajTrans(norm_method, self.mu, self.std)
            mu_std = torch.cat([self.mu.unsqueeze(0), self.std.unsqueeze(0)], dim=0)
            np.save("logs/mustd.npy", mu_std.cpu().numpy())
        self.transer = MomaTrajTrans(norm_method, self.mu, self.std)
        self.sample_matrix = None
        
        return
        # for uniform sampling
        unique_labels, counts = np.unique(self.min_idx, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))
        if min(label_counts.values()) == 0:
            print("数据集中存在空类别，请检查数据集")
            exit(1)

        self.sample_num = 5000
        sample_num = self.sample_num
        self.sample_matrix = torch.zeros(64, sample_num, dtype=torch.int64)
        for label in unique_labels:
            indices = np.where(self.min_idx == label)[0]  # 找到该类别的索引
            indices = list(indices)
            if len(indices) < sample_num:
                times = sample_num // len(indices)
                idxs = torch.as_tensor(indices).repeat(times)
                rand_num = sample_num % len(indices)
                sampled_indices = random.sample(list(indices), rand_num)
                idxs = torch.cat([idxs, torch.as_tensor(sampled_indices)], dim=0)
            else:
                idxs = torch.as_tensor(random.sample(indices, sample_num))
            self.sample_matrix[int(label)] = idxs.int()
        self.length_ = self.sample_matrix.shape[0] * sample_num

    def __len__(self):
        return self.length_
    
    def get_mustd_welford(self):
        chunk_size = 100
        # 初始化统计量
        n = 0  # 已处理样本数
        mu = np.zeros((self.wps_num, self.state_dim), dtype=np.float64)
        M2 = np.zeros((self.wps_num, self.state_dim), dtype=np.float64)
        
        for i in tqdm(range(0, self.length_, chunk_size)):
            # 读取当前块
            end = min(i + chunk_size, self.length_)
            chunk = self.wps[i:end]
            chunk = np.array(chunk, dtype=np.float64)  # 转换为float64
            tup = self.normer(torch.Tensor(chunk).cuda())
            chunk = tup[0].cpu().numpy()
            
            if chunk.size == 0:
                continue
            if len(chunk.shape) == 1:
                chunk = chunk.reshape(-1, 1)  # 确保二维
            
            k = chunk.shape[0]  # 当前块样本数
            sum_x = np.sum(chunk, axis=0)
            sum_x2 = np.sum(chunk ** 2, axis=0)
            
            # 计算当前块的均值和M2
            mu_k = sum_x / k
            M2_k = sum_x2 - (sum_x ** 2) / k
            
            # 合并到全局统计量
            if n == 0:
                mu = mu_k
                M2 = M2_k
                n = k
            else:
                new_n = n + k
                delta = mu_k - mu
                mu = (n * mu + k * mu_k) / new_n
                M2 += M2_k + (delta ** 2) * (n * k) / new_n
                n = new_n
        
        # 计算最终方差（总体方差）
        variance = M2 / n if n > 0 else np.zeros_like(M2)
        for i in range(self.wps_num):
            for j in range(self.state_dim):
                if variance[i, j] < 1.0e-6:
                    variance[i, j] = 1
        std = np.sqrt(variance)
        self.mu = torch.as_tensor(mu).float()
        self.std = torch.as_tensor(std).float()

    def __getitem__(self, idx):
        if self.sample_matrix is not None:
            # resample the idx
            mat_row = idx // self.sample_num
            mat_col = idx % self.sample_num
            idx = self.sample_matrix[mat_row, mat_col]
        
        occ_pos = torch.as_tensor(self.occs[idx]).float().contiguous()
        # boxes = torch.as_tensor(self.boxes[idx]).float().contiguous()
        sdf = torch.as_tensor(self.sdf[idx]).float().contiguous()
        tup = self.normer(torch.as_tensor(self.wps[idx]).float().contiguous().unsqueeze(0))
        wps = tup[0].squeeze(0)
        if tup[1] is None:
            startgoal = torch.cat([wps[0,-7:], wps[-1,:]])
        else:
            startgoal = torch.cat([wps[0,-7:], tup[1].squeeze(0), wps[-1,-9:]])
        wps_nor = (wps - self.mu) / self.std
        
        item_idx = torch.as_tensor(self.min_idx[idx]).long().contiguous()
        primitive = (self.trajlib[item_idx] - self.mu) / self.std
        data = {
            'pos': rotate_points(occ_pos, startgoal),
            'sdf': sdf,
            'x': wps_nor,
            'startgoal': startgoal,
            'min_idx': item_idx,
            'primitive': primitive,
            # 'primitive_loss': torch.tensor(self.primitive_loss[idx]),
            # 'boxes': norm_boxes(rotate_boxes(boxes, startgoal)),
            # 'raw_boxes' : boxes,
            'mu': self.mu,
            'std': self.std
        }
        
        # query key configurations
        if hasattr(self, "key_config"):
            key_config_free = isStateFree(self.key_config, sdf.flatten().unsqueeze(0), safe_margin=1e-3) * 1.0
            data['key_config_free'] = key_config_free
        
        return data
    
if __name__ == "__main__":
    ds = MomaDataset(filepath="train_nonfix.h5", 
    # ds = MomaDataset(filepath="data_train_raw.h5", 
                     norm_method="shoot", 
                     lib_path="logs/trajlib32.npy"
                     ,mustd_path="logs/mustd_nonfix.npy")
    
    # work on index
    idx_data = ds.min_idx
    idx_unique, idx_counts = np.unique(idx_data, return_counts=True)
    np.save("logs/class_counts.npy", idx_counts)
    max_count =np.percentile(idx_counts, 90, interpolation="nearest")
    min_count = np.percentile(idx_counts, 10, interpolation="nearest")
    print("max_count = ", max_count)
    print("min_count = ", min_count)

    plt.figure(figsize=(10, 6))
    plt.bar(idx_unique, idx_counts, color='skyblue')
    plt.xlabel('idx Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of idx')
    plt.xticks(idx_unique[::50], rotation=45)  # 每隔 50 个取值显示一个标签
    plt.grid(axis='y')
    plt.show()
    
    # work on trajlib
    trajlib = ds.trajlib
    trajlib_a = trajlib[:, 1:, :2]
    trajlib_b = trajlib[:, :-1, :2]
    trajlib_length = (trajlib_a - trajlib_b).norm(dim=-1).sum(dim=-1)
    plt.figure(figsize=(10, 6))
    plt.bar(idx_unique, trajlib_length, color='skyblue')
    np.save("logs/trajlib_length.npy", trajlib_length.numpy())
    plt.xlabel('idx Values')
    plt.ylabel('Traj Length')
    plt.title('Distribution of traj length')
    plt.xticks(idx_unique[::50], rotation=45)  # 每隔 50 个取值显示一个标签
    plt.grid(axis='y')
    plt.show()
    

    for i in range(1, len(ds)):
        
        # 获取第一个样本的数据
        data = ds[i]
        
        # boxes = data["boxes"]
        # print("wall boxes num = ", (boxes[:, 2] < 1e-3).sum().item())
        # print("float boxes num = ", (boxes[:, 2] > 1e-3).sum().item())
        
        occ_pos = data["pos"].cuda()
        wps = (data["x"].unsqueeze(0) * ds.std + ds.mu).cuda()  # denormalize the data
        wps = ds.denormer((wps, data["startgoal"][7:10].unsqueeze(0).cuda()))
        
        wps2 = ds.transer.detrans((data["x"].unsqueeze(0), data["startgoal"][7:10].unsqueeze(0)))
        print("tarnser diff = ", (wps - wps2.cuda()).abs().max().item())
        
        wps_data, target = ds.normer(wps)
        wps_raw = ds.denormer((wps_data, target))
        print("wps diff = ", (wps_raw - wps).abs().max().item())
        if target is not None:
            print("target diff = ", (data["startgoal"][7:10].unsqueeze(0).cuda() - target).abs().max().item())
        
        print(occ_pos.shape)
        print(wps.shape)
        
        idx = data["min_idx"].item()
        print("min_idx = ", idx)
        idx_traj = data["primitive"].cuda()
        idx_traj = ds.transer.detrans((idx_traj.unsqueeze(0), data["startgoal"][7:10].unsqueeze(0).cuda()))
        wps = torch.cat([wps, idx_traj], dim=0)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(occupied_points_world[:, 0], occupied_points_world[:, 1], occupied_points_world[:, 2], c='black', marker='o')
        occ_posn = occ_pos.cpu().numpy()
        ax.scatter(occ_posn[:, 0], occ_posn[:, 1], occ_posn[:, 2], c='red', marker='o')
        points_wps_cuda, _ = get_robo_tensor(wps.cuda())
        # 绘制 wps
        for i in range(2):
            points_wps = points_wps_cuda.cpu().numpy()[i]
            for j in range(points_wps.shape[0]):
                color = plt.cm.coolwarm(j / points_wps.shape[0])  # 使用冷暖渐变颜色
                if i == 1:
                    color = 'black'
                ax.plot(points_wps[j, :, 0], points_wps[j, :, 1], points_wps[j, :, 2], c=color, marker='o')
                               
        ax.set_aspect('equal', 'box')
        manager = plt.get_current_fig_manager()
        manager.resize(2048, 1660)
        def on_key_event(event):
            if event.key == 'q':
                plt.close()
                sys.exit()
        plt.gcf().canvas.mpl_connect('key_press_event', on_key_event)
        plt.show()
