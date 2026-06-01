#!/usr/bin/env python3
from utils.data_loader import MomaDataset
import numpy as np
from tqdm import tqdm
from utils.utils import *
from loss import *
from einops import rearrange

import h5py

def safe_loss(traj, data):
    sdf = data["sdf"].detach().unsqueeze(0).flatten(start_dim=1).to(traj.device)
    dist =  - getDistanceFromState(traj.reshape(-1, 11), sdf) + colli_point_radius.to(traj.device)
    cost = positiveSmoothedL2(dist.flatten()).mean()
    return cost

def get_points(fake_batch):
    # b x wpsnum x 11 -> b x wps_num x 9 x 3
        
    dof_num = 7
    moma_pos_batch = fake_batch.view(-1, dof_num+4)
    batch_size = fake_batch.size(0)
    fxxk_size = batch_size * fake_batch.size(1)
    zeros_cuda = torch.zeros(fxxk_size, device=fake_batch.device)
    ones_cuda = torch.ones(fxxk_size, device=fake_batch.device)
    
    # 初始化位置和旋转矩阵
    now_p = torch.stack([moma_pos_batch[:, 0], moma_pos_batch[:, 1], ones_cuda * chassis_height], dim=1)
    now_R = torch.stack([
        torch.stack([moma_pos_batch[:, 2], -moma_pos_batch[:, 3], zeros_cuda], dim=1),
        torch.stack([moma_pos_batch[:, 3], moma_pos_batch[:, 2], zeros_cuda], dim=1),
        torch.stack([zeros_cuda, zeros_cuda, ones_cuda], dim=1)
    ], dim=1)
    
    first_point = torch.cat([now_p.clone(), moma_pos_batch[:, 2].unsqueeze(1), ones_cuda.unsqueeze(1)], dim=1)
    colli_pts = [first_point.unsqueeze(1)]
    now_p += torch.matmul(now_R, relative_t.to(fake_batch.device))
    now_R = torch.matmul(now_R, relative_R.to(fake_batch.device))
    second_point = torch.cat([now_p.clone(), moma_pos_batch[:, 3].unsqueeze(1), 2.0*ones_cuda.unsqueeze(1)], dim=1)
    colli_pts.append(second_point.unsqueeze(1))

    for i in range(dof_num):
        now_p += now_R[:, :, 2] * colli_length[i]
        now_point = torch.cat([now_p.clone(), moma_pos_batch[:, 4+i].unsqueeze(1), (3.0+i)*ones_cuda.unsqueeze(1)], dim=1)
        colli_pts.append(now_point.unsqueeze(1).clone())
        pn = 1.0
        if i % 2 == 0:
            dof_R = torch.stack([
                torch.stack([torch.cos(pn * moma_pos_batch[:, 4 + i]), -torch.sin(pn * moma_pos_batch[:, 4 + i]), zeros_cuda], dim=1),
                torch.stack([torch.sin(pn * moma_pos_batch[:, 4 + i]), torch.cos(pn * moma_pos_batch[:, 4 + i]), zeros_cuda], dim=1),
                torch.stack([zeros_cuda, zeros_cuda, ones_cuda], dim=1)
            ], dim=1)
        else:
            dof_R = torch.stack([
                torch.stack([torch.cos(pn * moma_pos_batch[:, 4 + i]), zeros_cuda, torch.sin(pn * moma_pos_batch[:, 4 + i])], dim=1),
                torch.stack([zeros_cuda, ones_cuda, zeros_cuda], dim=1),
                torch.stack([-torch.sin(pn * moma_pos_batch[:, 4 + i]), zeros_cuda, torch.cos(pn * moma_pos_batch[:, 4 + i])], dim=1)
            ], dim=1)
        now_R = torch.matmul(now_R, dof_R)

    all_colli_pts = torch.cat(colli_pts, dim=1)
    all_colli_pts = rearrange(all_colli_pts, '(b c) h w -> b c h w', b=batch_size, c=fake_batch.size(1))
    
    return all_colli_pts[..., :3]

# def find_min_idx(now_path, trajlib, data, topnum = 3):
#     # uni_path: wps_num*state_dim
#     # safe_path: M*wps_num*state_dim
#     path = now_path.unsqueeze(0).repeat(trajlib.shape[0], 1, 1)
#     distances = (path - trajlib).norm(dim=2)
#     distances = -distances.mean(dim=1)
#     _, dists_idxs = torch.topk(distances, topnum)
    
#     min_idx = dists_idxs[0]
#     safe_cost = safe_loss(trajlib[min_idx], data)
#     for i in range(1, topnum):
#         idx = dists_idxs[i]
#         idx_safe_cost = safe_loss(trajlib[idx], data)
#         if idx_safe_cost < safe_cost:
#             min_idx = idx
#             safe_cost = idx_safe_cost
#     return min_idx

# def find_min_idx(now_path, trajlib):
#     # uni_path: wps_num*state_dim
#     # safe_path: M*wps_num*state_dim
#     path = now_path.unsqueeze(0).repeat(trajlib.shape[0], 1, 1)
#     distances = (path - trajlib).norm(dim=2)
#     return torch.argmin(distances.mean(dim=1))

def find_min_idx(data, trajlib_points, transer):
    # now_path: 1 x wps_num x 11
    # safe_path: M x wps_num x 9 x 3
    
    # wps = transer.detrans((data["x"].unsqueeze(0), data["startgoal"][7:10].unsqueeze(0)))
    wps = data["x"] * transer.std + transer.mu
    wps = wps.unsqueeze(0)
    
    wps = wps.to(trajlib.device)
    points_path = get_points(wps)
    
    # cuda_start_goal = data["startgoal"][7:10].to(trajlib.device).unsqueeze(0).repeat(trajlib.shape[0], 1)
    # trajlib_points = transer.denormer((trajlib, cuda_start_goal))
    # trajlib_points = get_points(trajlib_points)
    
    
    path = points_path.repeat(trajlib_points.shape[0], 1, 1, 1)
    distances = (path - trajlib_points).norm(dim=3).mean(dim=2)
    return torch.argmin(distances.mean(dim=1))

def find_min_idx_se2(data, trajlib, transer):
    """retuns the matching index of traj in trajlib

    Args:
        traj (tensor): 64x11
        trajlib (_type_): traj
        transer (_type_): _description_
    """
    # assert trajlib.shape[-1] == 4, "trajlib should be in SE(2) format"
    wps = data["x"] * transer.std + transer.mu
    wps = rearrange(wps, 'w x -> 1 w x')
    wps = wps.to(trajlib.device)
    
    wps_se2 = wps[..., :4] # (1x64x4)
    distances = (wps_se2 - trajlib[..., :4]) # (32 x 64 x 4)
    
    distances = distances.norm(dim=-1)
    distances = distances.mean(dim=-1)
    return torch.argmin(distances)

def l2loss(data, trajlib, transer):
    """retuns the matching index of traj in trajlib

    Args:
        traj (tensor): 64x11
        trajlib (_type_): traj
        transer (_type_): _description_
    """
    # assert trajlib.shape[-1] == 4, "trajlib should be in SE(2) format"
    wps = data["x"] * transer.std + transer.mu
    wps = rearrange(wps, 'w x -> 1 w x')
    wps = wps.to(trajlib.device)
    
    wps_se2 = wps[..., :2] # (1x64x4)
    distances = (wps_se2 - trajlib[..., :2]) # (32 x 64 x 4)
    
    distances = distances.norm(dim=-1)
    distances, _ = distances.max(dim=-1)
    # distances = distances.mean(dim=-1)
    return distances

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str)
    parser.add_argument('-o', '--output_path', type=str, default='./trajlib_loss.h5')
    parser.add_argument('--norm_method', type=str, default="shoot")
    parser.add_argument('--lib_path', type=str, default="logs/trajlib32.npy")
    args = parser.parse_args()
    
    ds = MomaDataset(filepath=args.filepath, 
                    norm_method=args.norm_method,
                    lib_path=args.lib_path)
    print(ds[0]['x'].shape)
    trajlib = np.load(args.lib_path, allow_pickle=True)
    trajlib = torch.as_tensor(trajlib).float().cuda()
    print("trajlib shape:", trajlib.shape)
    traj_num = trajlib.shape[0]


    f = h5py.File(args.output_path, 'w')
    l2 = f.create_dataset('l2_loss',
                shape=(len(ds), traj_num),
                dtype=np.float32)
    
    for i in tqdm(range(len(ds))):
        data = ds[i]
        loss = l2loss(data, trajlib, ds.transer)
        l2[i] = loss.cpu().numpy()

    f.close()

