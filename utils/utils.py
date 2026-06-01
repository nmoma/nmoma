import torch.nn as nn
from moma_param import MomaParam
import math
import numpy as np
import torch
from torch import nn
from torch import einsum
import torch.nn.functional as F
from einops import repeat, rearrange
from inspect import isfunction
from utils.rotation_conversions import *

try:
    import flash_attn
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    flash_attn = None

MOMA_PARAM = MomaParam()
map_size_2d = (100, 100, )
map_size_3d = (100, 100, 16, )
anchor_shape = (10, 10, 4, )
map_resolution = 0.1
map_resolution_inv = 1.0 / map_resolution
anchor_resolution = torch.tensor([map_size_3d[0] / anchor_shape[0] * map_resolution, 
                                map_size_3d[1] / anchor_shape[1] * map_resolution, 
                                map_size_3d[2] / anchor_shape[2] * map_resolution])
anchor_resolution_inv = torch.tensor([anchor_shape[0] / map_size_3d[0] * map_resolution_inv, 
                                    anchor_shape[1] / map_size_3d[1] * map_resolution_inv, 
                                    anchor_shape[2] / map_size_3d[2] * map_resolution_inv])
map_origin = [-5.0, -5.0, 0.0]
map2d_origin_tensor = torch.tensor(map_origin[:2])
map3d_origin_tensor = torch.tensor(map_origin)

robot_param = MomaParam()
min_joint = robot_param.getJointLimitsMin()
max_joint = robot_param.getJointLimitsMax()
joint_vel_limit = robot_param.getJointVelLimits()

chassis_height = robot_param.chassis_height
max_v = robot_param.max_v
max_a = robot_param.max_a
max_w = robot_param.max_w
max_dw = robot_param.max_dw
colli_length = robot_param.colli_length
colli_points = robot_param.colli_points

raw_colli_radius = robot_param.colli_point_radius
colli_point_radius = []
for i in range(len(raw_colli_radius)):
    if raw_colli_radius[i] > 1e-3:
        colli_point_radius.append(raw_colli_radius[i])
colli_point_radius.append(robot_param.chassis_colli_radius)
colli_point_radius = torch.as_tensor(colli_point_radius).float()

relative_R = torch.as_tensor(robot_param.relative_R).float()
relative_t = torch.as_tensor(robot_param.relative_t).float()

# print("colli_length: ", colli_length)
# print("colli_points: ", colli_points)
# print("colli_point_radius: ", colli_point_radius)
# print("relative_R: ", relative_R)
# print("relative_t: ", relative_t)
# print("max_joint: ", max_joint)

def expC2(tensor):
    idx_ge = tensor >= 0.0
    idx_lt = tensor < 0.0
    tensor[idx_ge] = (0.5 * tensor[idx_ge] + 1.0) * tensor[idx_ge] + 1.0
    tensor[idx_lt] = (0.5 * tensor[idx_lt] - 1.0) * tensor[idx_lt] + 1.0
    return tensor

def logC2(tensor):
    idx_ge = tensor >= 1.0
    idx_lt = tensor < 1.0
    tensor[idx_ge] = torch.sqrt(2.0 * (tensor[idx_ge] - 1.0))
    tensor[idx_lt] = 1.0 - torch.sqrt(2.0 / tensor[idx_lt] - 1.0)
    return tensor

def sigmoidC2(tensor, max_tensor):
    # tensor: ..., N
    # max_tensor: ..., N
    tensor = expC2(tensor)
    return 2.0 * max_tensor * tensor / (1.0 + tensor) - max_tensor

def invSigmoidC2(tensor, max_tensor):
    # tensor: ..., N
    # max_tensor: ..., N
    b = 0.5 * (tensor + max_tensor) / max_tensor
    return logC2(b/(1-b))

def norm_trajs_s(trajs_in):
    # b, wpsnum, 11
    trajs = trajs_in.clone()
    trajs[..., 0] /= 5.0
    trajs[..., 1] /= 5.0
    trajs[..., 4:11] = trajs[..., 4:11] / torch.as_tensor(max_joint).to(trajs_in.device)
    return trajs, None

def denorm_trajs_s(data):
    # b, wpsnum, 11
    trajs = data[0].clone()
    trajs[..., 0] *= 5.0
    trajs[..., 1] *= 5.0
    trajs[..., 4:11] = trajs[..., 4:11] * torch.as_tensor(max_joint).to(data[0].device)
    return trajs

def to_pspace_s(startgoals):
    # b, 7+11
    starts = startgoals[:, :7]
    goals = startgoals[:, 7:]
    full_sg = torch.cat([torch.Tensor([[0.0, 0.0, 1.0, 0.0]]).to(startgoals.device)
                            .repeat(starts.shape[0], 1), starts, goals], dim=1)
    full_sg = full_sg.reshape(starts.shape[0], 2, 11)
    return get_box_tensor(denorm_trajs_s((full_sg,None))).to(startgoals.dtype)
    # return get_point_tensor(denorm_trajs_s((full_sg,None)))
    
def classify_points(traj, class_num = 36):
    # ...x>11
    # output: ... x11
    if (traj<-1.0).any() or (traj>1.0).any():
        # print("Warning: traj out of range:")
        # print(traj)
        traj = torch.clamp(traj, -1.0, 1.0)
    normed_traj = traj.clone()
    # assert (normed_traj>=-1.0).all() and (normed_traj <=1.0).all()
    normed_traj = normed_traj / 2.0 + 0.5
    ret = torch.zeros_like(normed_traj)
    for i in range(0, class_num):
        choose = (normed_traj >= (i/class_num)) & (normed_traj < ((i+1)/class_num))
        ret += i * choose
    return ret

def norm_trajs(trajs_in):
    # b, wpsnum, 11
    trajs = trajs_in.clone()
    targets = trajs[:, -1, :2].clone()
    local_diff = targets - trajs[:, 0, :2]
    normalizeL = torch.norm(local_diff, dim=1, keepdim=True)
    angle = torch.atan2(local_diff[:, 1], local_diff[:, 0]).unsqueeze(1).unsqueeze(2)
    sangle = torch.sin(angle)
    cangle = torch.cos(angle)
    R = torch.cat([torch.cat([cangle, -sangle], dim=2),
                    torch.cat([sangle, cangle], dim=2)], dim=1)
    trajs[:, :, :2] = (trajs[:, :, :2] @ R) / normalizeL.unsqueeze(2)
    sangle = sangle.squeeze(2)
    cangle = cangle.squeeze(2)
    sdiff = trajs[:, :, 3]*cangle - trajs[:, :, 2]*sangle
    cdiff = trajs[:, :, 3]*sangle + trajs[:, :, 2]*cangle
    trajs[:, :, 2] = cdiff
    trajs[:, :, 3] = sdiff
    trajs[..., 4:11] = trajs[..., 4:11] / torch.as_tensor(max_joint).to(trajs_in.device)
    return trajs, torch.cat([normalizeL, cangle, sangle], dim=1)

def denorm_trajs(trajs_in, targets_in):
    # trajs: b, wpsnum, 11
    # targets: b, 3
    trajs = torch.zeros_like(trajs_in)
    targets = targets_in.clone()
    cangle = targets[:, 1].unsqueeze(1)
    sangle = targets[:, 2].unsqueeze(1)
    trajs[:, :, 2] = trajs_in[:, :, 2] * cangle - trajs_in[:, :, 3] * sangle
    trajs[:, :, 3] = trajs_in[:, :, 3] * cangle + trajs_in[:, :, 2] * sangle
    sangle = sangle.unsqueeze(2)
    cangle = cangle.unsqueeze(2)
    normalizeL = targets[:, :1].unsqueeze(2)
    R = torch.cat([torch.cat([cangle, sangle], dim=2),
                    torch.cat([-sangle, cangle], dim=2)], dim=1)
    trajs[:, :, :2] = (trajs_in[:, :, :2] * normalizeL) @ R
    trajs[..., 4:11] = trajs_in[..., 4:11] * torch.as_tensor(max_joint, device=trajs_in.device)
    return trajs

def denorm_trajs_data(data):
    # trajs: b, wpsnum, 11
    # targets: b, 3
    return denorm_trajs(data[0], data[1])

def to_pspace(startgoals):
    # b, 7+3+9
    starts = startgoals[:, :7]
    target = startgoals[:, 7:10]
    goals = startgoals[:, 10:]
    full_sg = torch.cat([torch.Tensor([[0.0, 0.0]]).to(startgoals.device).repeat(starts.shape[0], 1), target[:, 1:], starts, 
                         torch.Tensor([[1.0, 0.0]]).to(startgoals.device).repeat(starts.shape[0], 1), goals], dim=1).half()
    return get_point_tensor_pure(denorm_trajs(full_sg.reshape(starts.shape[0], 2, 11), target))

def to_pspace_pspace(startgoals):
    # b, 7+3+9
    starts = startgoals[:, :7]
    target = startgoals[:, 7:10]
    goals = startgoals[:, 10:]
    full_sg = torch.cat([torch.Tensor([[0.0, 0.0]]).to(startgoals.device).repeat(starts.shape[0], 1), target[:, 1:], starts, 
                         torch.Tensor([[1.0, 0.0]]).to(startgoals.device).repeat(starts.shape[0], 1), goals], dim=1).half()
    return get_point_tensor(denorm_trajs(full_sg.reshape(starts.shape[0], 2, 11), target))


NORMER = {
    "simple": norm_trajs_s,
    "shoot": norm_trajs
}

DENORMER = {
    "simple": denorm_trajs_s,
    "shoot": denorm_trajs_data
}

SPACER = {
    "simple": to_pspace_s,
    "shoot": to_pspace,
    "shoot_pspace": to_pspace_pspace
}

def filter(opState, kernelsize=5):
    #B*wpsnum*state_dim

    Bs = opState.shape[0]
    ches = opState.shape[1]
    recL = int((kernelsize-3)/2)
    labelTable = torch.zeros(Bs, int(ches+2*recL),opState.shape[2]).cuda()
    labelTable[:,:recL,:] =   opState[:,0,:].unsqueeze(dim=1)
    labelTable[:,-recL:,:] =   opState[:,-1,:].unsqueeze(dim=1)
    labelTable[:,recL:-recL,:] = opState

    newOpState = torch.zeros_like(opState)

    tmpT = labelTable.unfold(1, kernelsize, 1)
    tmpMeanT = torch.mean(tmpT, dim=-1)
    newOpState[:,1:-1,:] = tmpMeanT
    newOpState[:,0,:] = opState[:,0,:]
    newOpState[:,-1,:] = opState[:,-1,:]

    return newOpState

def map2idx2d(map_coords):
    idx_coords = torch.floor((map_coords[..., :2] - map2d_origin_tensor.cuda()) * map_resolution_inv)
    idx_coords = idx_coords.long()
    return idx_coords

def idx2map2d(idx_coords):
    map_coords = (idx_coords.float() + 0.5) * map_resolution + map2d_origin_tensor
    return map_coords

def map2idx3d(map_coords):
    idx_coords = torch.floor((map_coords[..., :3] - map3d_origin_tensor.cuda()) * map_resolution_inv)
    idx_coords = idx_coords.long()
    return idx_coords

def map2idx3dAnchor(map_coords):
    idx_coords = torch.floor((map_coords[..., :3] - map3d_origin_tensor.cuda()) * anchor_resolution_inv.cuda())
    idx_coords = idx_coords.long()
    return idx_coords

def idx2map3d(idx_coords):
    map_coords = (idx_coords.float() + 0.5) * map_resolution + map3d_origin_tensor.to(idx_coords.device)
    return map_coords

def idx2map3dLow(idx_coords):
    map_coords = idx_coords.float() * map_resolution + map3d_origin_tensor
    return map_coords

def idx2map3dAnchor(idx_coords):
    map_coords = (idx_coords + 0.5) * anchor_resolution.cuda() + map3d_origin_tensor.cuda()
    return map_coords

def idx2map3dAnchorLow(idx_coords):
    map_coords = idx_coords* anchor_resolution.cuda() + map3d_origin_tensor.cuda()
    return map_coords

def boundAnchorIdx(idx_coords):
    idx_coords[..., 0] = np.clip(idx_coords[..., 0], 0, anchor_shape[0]-1)
    idx_coords[..., 1] = np.clip(idx_coords[..., 1], 0, anchor_shape[1]-1)
    idx_coords[..., 2] = np.clip(idx_coords[..., 2], 0, anchor_shape[2]-1)
    return idx_coords

def get_se2_colli_tensor(se2_pos_batch):
    batch_size = se2_pos_batch.size(0)
    colli_pts_tensor = torch.zeros((batch_size, 3, *map_size_2d)).cuda()
    idx_coords = map2idx2d(se2_pos_batch)
    colli_pts_tensor[:, 0, idx_coords[:, 0], idx_coords[:, 1]] = 1
    colli_pts_tensor[:, 1, idx_coords[:, 0], idx_coords[:, 1]] = torch.cos(se2_pos_batch[:, 2])
    colli_pts_tensor[:, 2, idx_coords[:, 0], idx_coords[:, 1]] = torch.sin(se2_pos_batch[:, 2])
    return colli_pts_tensor

def get_fk_pose(qin):
    # b x 11 -> b x (3+6)
    
    dof_num = 7
    batch_size = qin.size(0)
    zeros_cuda = torch.zeros(batch_size, device=qin.device)
    ones_cuda = torch.ones(batch_size, device=qin.device)
    
    # 初始化位置和旋转矩阵
    now_p = torch.stack([qin[:, 0], qin[:, 1], ones_cuda * chassis_height], dim=1)
    now_R = torch.stack([
        torch.stack([qin[:, 2], -qin[:, 3], zeros_cuda], dim=1),
        torch.stack([qin[:, 3], qin[:, 2], zeros_cuda], dim=1),
        torch.stack([zeros_cuda, zeros_cuda, ones_cuda], dim=1)
    ], dim=1)
    
    now_p += torch.matmul(now_R, relative_t.to(qin.device))
    now_R = torch.matmul(now_R, relative_R.to(qin.device))

    for i in range(dof_num):
        now_p += now_R[:, :, 2] * colli_length[i]
        pn = 1.0
        if i % 2 == 0:
            dof_R = torch.stack([
                torch.stack([torch.cos(pn * qin[:, 4 + i]), -torch.sin(pn * qin[:, 4 + i]), zeros_cuda], dim=1),
                torch.stack([torch.sin(pn * qin[:, 4 + i]), torch.cos(pn * qin[:, 4 + i]), zeros_cuda], dim=1),
                torch.stack([zeros_cuda, zeros_cuda, ones_cuda], dim=1)
            ], dim=1)
        else:
            dof_R = torch.stack([
                torch.stack([torch.cos(pn * qin[:, 4 + i]), zeros_cuda, torch.sin(pn * qin[:, 4 + i])], dim=1),
                torch.stack([zeros_cuda, ones_cuda, zeros_cuda], dim=1),
                torch.stack([-torch.sin(pn * qin[:, 4 + i]), zeros_cuda, torch.cos(pn * qin[:, 4 + i])], dim=1)
            ], dim=1)
        now_R = torch.matmul(now_R, dof_R)

    now_R = matrix_to_rotation_6d(now_R)
    return torch.cat([now_p, now_R], dim=1)

def get_robo_tensor(fake_batch, requires_grad=False):
    # b x wpsnum x 11 -> b x wpsnum x 9 x 3
    if requires_grad:
        fake_batch = fake_batch.requires_grad_(True)
        
    dof_num = 7
    moma_pos_batch = fake_batch.view(-1, dof_num+4)
    batch_size = fake_batch.size(0)
    fxxk_size = batch_size * fake_batch.size(1)
    zeros_cuda = torch.zeros(fxxk_size, device=fake_batch.device, requires_grad=requires_grad)
    ones_cuda = torch.ones(fxxk_size, device=fake_batch.device, requires_grad=requires_grad)
    
    # 初始化位置和旋转矩阵
    now_p = torch.stack([moma_pos_batch[:, 0], moma_pos_batch[:, 1], ones_cuda * chassis_height], dim=1)
    now_R = torch.stack([
        torch.stack([moma_pos_batch[:, 2], -moma_pos_batch[:, 3], zeros_cuda], dim=1),
        torch.stack([moma_pos_batch[:, 3], moma_pos_batch[:, 2], zeros_cuda], dim=1),
        torch.stack([zeros_cuda, zeros_cuda, ones_cuda], dim=1)
    ], dim=1)
    
    colli_pts = [now_p.unsqueeze(1).clone()]
    now_p += torch.matmul(now_R, relative_t.to(fake_batch.device))
    now_R = torch.matmul(now_R, relative_R.to(fake_batch.device))
    colli_pts.append(now_p.unsqueeze(1).clone())

    for i in range(dof_num):
        now_p += now_R[:, :, 2] * colli_length[i]
        colli_pts.append(now_p.unsqueeze(1).clone())
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
    
    if not requires_grad:
        idx_coords = map2idx3d(all_colli_pts)
        return all_colli_pts, idx_coords
    
    return all_colli_pts

def get_point_tensor_pure(fake_batch):
    # b x 2 x 11 -> b x (2x9) x 4
        
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
    
    first_point = torch.cat([now_p.clone(), moma_pos_batch[:, 2].unsqueeze(1)], dim=1)
    colli_pts = [first_point.unsqueeze(1)]
    now_p += torch.matmul(now_R, relative_t.to(fake_batch.device))
    now_R = torch.matmul(now_R, relative_R.to(fake_batch.device))
    second_point = torch.cat([now_p.clone(), moma_pos_batch[:, 3].unsqueeze(1)], dim=1)
    colli_pts.append(second_point.unsqueeze(1))

    for i in range(dof_num):
        now_p += now_R[:, :, 2] * colli_length[i]
        now_point = torch.cat([now_p.clone(), moma_pos_batch[:, 4+i].unsqueeze(1)], dim=1)
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
    
    return rearrange(all_colli_pts, '(b c) h w -> b (c h) w', b=batch_size, c=fake_batch.size(1))

def get_point_tensor(fake_batch):
    # b x 2 x 11 -> b x (2x9) x (3+5)
        
    dof_num = 7
    moma_pos_batch = fake_batch.view(-1, dof_num+4)
    batch_size = fake_batch.size(0)
    fxxk_size = batch_size * fake_batch.size(1)
    zeros_cuda = torch.zeros(fxxk_size, device=fake_batch.device)
    ones_cuda = torch.ones(fxxk_size, device=fake_batch.device)
    ozz = torch.Tensor([1., 0., 0.]).to(fake_batch.device).repeat(batch_size, 1, 9, 1)
    zoz = torch.Tensor([0., 1., 0.]).to(fake_batch.device).repeat(batch_size, 1, 9, 1)
    
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
    start_all = torch.cat([all_colli_pts[:, :1], ozz], dim=-1)
    goal_all = torch.cat([all_colli_pts[:, 1:], zoz], dim=-1)
    all_points = rearrange(torch.cat([start_all, goal_all], dim=1), 'b c h w -> b (c h) w')
    
    return all_points

def get_box_tensor(fake_batch):
    # b x 2 x 11 -> b x (2x9) x (8+3)
    dof_num = 7
    moma_pos_batch = fake_batch.view(-1, dof_num+4)
    batch_size = fake_batch.size(0)
    fxxk_size = batch_size * fake_batch.size(1)
    zeros_cuda = torch.zeros(fxxk_size, device=fake_batch.device)
    ones_cuda = torch.ones(fxxk_size, device=fake_batch.device)
    ozz = torch.Tensor([1., 0., 0.]).to(fake_batch.device).repeat(batch_size, 1, 9, 1)
    zoz = torch.Tensor([0., 1., 0.]).to(fake_batch.device).repeat(batch_size, 1, 9, 1)
    msize = -torch.ones((fxxk_size, 3), device=fake_batch.device)
    
    # 初始化位置和旋转矩阵
    now_p = torch.stack([moma_pos_batch[:, 0], moma_pos_batch[:, 1], ones_cuda * chassis_height], dim=1)
    now_R = torch.stack([
        torch.stack([moma_pos_batch[:, 2], -moma_pos_batch[:, 3], zeros_cuda], dim=1),
        torch.stack([moma_pos_batch[:, 3], moma_pos_batch[:, 2], zeros_cuda], dim=1),
        torch.stack([zeros_cuda, zeros_cuda, ones_cuda], dim=1)
    ], dim=1)
    
    first_point = torch.cat([now_p.clone(), msize,
                             moma_pos_batch[:, 2].unsqueeze(1),
                             ones_cuda.unsqueeze(1)], dim=1)
    colli_pts = [first_point.unsqueeze(1)]
    now_p += torch.matmul(now_R, relative_t.to(fake_batch.device))
    now_R = torch.matmul(now_R, relative_R.to(fake_batch.device))
    second_point = torch.cat([now_p.clone(), msize, moma_pos_batch[:, 3].unsqueeze(1), 2.0*ones_cuda.unsqueeze(1)], dim=1)
    colli_pts.append(second_point.unsqueeze(1))

    for i in range(dof_num):
        now_p += now_R[:, :, 2] * colli_length[i]
        now_point = torch.cat([now_p.clone(), msize, moma_pos_batch[:, 4+i].unsqueeze(1), (3.0+i)*ones_cuda.unsqueeze(1)], dim=1)
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
    start_all = torch.cat([all_colli_pts[:, :1], ozz], dim=-1)
    goal_all = torch.cat([all_colli_pts[:, 1:], zoz], dim=-1)
    all_points = rearrange(torch.cat([start_all, goal_all], dim=1), 'b c h w -> b (c h) w')
    
    return all_points

def unrotate_points(rotated_pos, startgoals):
    # rotated_pos: Nx3 / BxNx3 - points that have been rotated
    # startgoals: contains sin_r and cos_r values for rotation
    
    is_batch_ = False if rotated_pos.dim() == 2 else True
    
    pos_ = rotated_pos.reshape(-1, 3) if is_batch_ else rotated_pos
    
    print(startgoals.shape)
    if is_batch_ : print("is_batch")
    # Get the same cos_r and sin_r values used in original rotation
    cos_r = startgoals[:, 8].repeat_interleave(rotated_pos.shape[1], dim=0) if is_batch_ else startgoals[8]
    sin_r = startgoals[:, 9].repeat_interleave(rotated_pos.shape[1], dim=0) if is_batch_ else startgoals[9]
    
    # Create inverse rotation matrices (transpose of original rotation matrices)
    inverse_rotation_matrices = torch.zeros((pos_.shape[0], 3, 3))
    inverse_rotation_matrices[:, 0, 0] = cos_r
    inverse_rotation_matrices[:, 0, 1] = sin_r  # Note: sign changed from original
    inverse_rotation_matrices[:, 1, 0] = -sin_r  # Note: sign changed from original  
    inverse_rotation_matrices[:, 1, 1] = cos_r
    inverse_rotation_matrices[:, 2, 2] = 1
    
    return torch.bmm(pos_.unsqueeze(1), inverse_rotation_matrices).squeeze(1).reshape(rotated_pos.shape)

def rotate_points(pos, startgoals):
    # boxes:  Nx3 / BxNx3
    # rad:    N / BxN

    is_batch_ = False if pos.dim() == 2 else True

    pos_ = pos.reshape(-1, 3) if is_batch_ else pos

    cos_r = startgoals[:, 8].repeat_interleave(pos.shape[1], dim=0) if is_batch_ else startgoals[8]
    sin_r = startgoals[:, 9].repeat_interleave(pos.shape[1], dim=0) if is_batch_ else startgoals[9]

    rotation_matrices = torch.zeros((pos_.shape[0], 3, 3))
    rotation_matrices[:, 0, 0] = cos_r
    rotation_matrices[:, 0, 1] =-sin_r
    rotation_matrices[:, 1, 0] = sin_r
    rotation_matrices[:, 1, 1] = cos_r
    rotation_matrices[:, 2, 2] = 1

    return torch.bmm(pos_.unsqueeze(1), rotation_matrices).squeeze(1).reshape(pos.shape)

def rotate_boxes(boxes, startgoals):
    # boxes:  Nx8 / BxNx8
    # rad:    N / BxN

    devices = boxes.device

    is_batch_ = False if boxes.dim() == 2 else True
    # rad_ = rad if torch.is_tensor(rad) else torch.tensor(rad)
    # if len(rad_.shape) == 0 : rad_ = rad_.reshape((1,))

    boxes_ = boxes.reshape(-1, 8) if is_batch_ else boxes

    # if is_batch_ : assert boxes.shape[0] == rad_.shape[0]
    # if is_batch_ : rad_ = rad_.repeat_interleave(boxes.shape[1])

    x, y, z, w, l, h, cos_theta, sin_theta = boxes_.unbind(dim=1)

    cos_r = startgoals[:, 8].repeat_interleave(boxes.shape[1], dim=0) if is_batch_ else startgoals[8]
    sin_r = startgoals[:, 9].repeat_interleave(boxes.shape[1], dim=0) if is_batch_ else startgoals[9]

    new_cos_theta = cos_theta * cos_r + sin_theta * sin_r
    new_sin_theta = sin_theta * cos_r - cos_theta * sin_r

    vertices = boxes_[:, :3].unsqueeze(1) # Mx1x3


    rotation_matrices = torch.zeros((boxes_.shape[0], 3, 3))
    rotation_matrices[:, 0, 0] = cos_r
    rotation_matrices[:, 0, 1] =-sin_r
    rotation_matrices[:, 1, 0] = sin_r
    rotation_matrices[:, 1, 1] = cos_r
    rotation_matrices[:, 2, 2] = 1

    rotated_vertices = torch.bmm(vertices, rotation_matrices)

    # rotated_vertices = torch.matmul(vertices, rotation_matrix.T)

    new_boxes = torch.cat([rotated_vertices.squeeze(1), 
                           boxes_[:, 3:-2], 
                           new_cos_theta.reshape(-1,1), 
                           new_sin_theta.reshape(-1,1)], dim=1)
    return new_boxes.reshape(boxes.shape)

def norm_boxes(boxes):
    # boxes: (N, 8) or (B, N, 8)
    # Returns normalized boxes in the same shape
    
    # if boxes.dim() == 2:
    #     boxes = boxes.unsqueeze(0)  # Add batch dim if missing
    # B, N, _ = boxes.shape
    
    # Extract components

    is_batch = True if boxes.dim() == 3 else False

    x = boxes[..., 0]
    y = boxes[..., 1]
    z = boxes[..., 2]
    w = boxes[..., 3]
    l = boxes[..., 4]
    h = boxes[..., 5]
    cos_theta = boxes[..., 6]
    sin_theta = boxes[..., 7]
    
    # Compute theta and normalize to [0, 2π)
    theta = torch.atan2(sin_theta, cos_theta)
    theta = theta % (2 * math.pi)
    
    # Determine quadrant (0, 1, 2, 3)
    quadrant = (theta / (0.5 * math.pi)).long() % 4
    
    # Normalize theta to [0, 0.5π)
    normalized_theta = theta % (0.5 * math.pi)
    new_cos_theta = torch.cos(normalized_theta)
    new_sin_theta = torch.sin(normalized_theta)
    
    # Swap w and l if in quadrant 1 or 3
    new_w = torch.where((quadrant == 1) | (quadrant == 3), l, w)
    new_l = torch.where((quadrant == 1) | (quadrant == 3), w, l)
    
    # Adjust x and y based on quadrant to ensure rotation about the new vertex
    new_x = torch.where(
        quadrant == 0, x,
        torch.where(
            quadrant == 1, x - l * sin_theta,
            torch.where(
                quadrant == 2, x + w * cos_theta - l * sin_theta,
                x + w * cos_theta
            )
        )
    )
    
    new_y = torch.where(
        quadrant == 0, y,
        torch.where(
            quadrant == 1, y + l * cos_theta,
            torch.where(
                quadrant == 2, y + w * sin_theta + l * cos_theta,
                y + w * sin_theta
            )
        )
    )
    
    theta -= (math.pi / 2) * quadrant.float()
    new_cos_theta = torch.cos(theta)
    new_sin_theta = torch.sin(theta)

    # Construct normalized boxes
    norm_boxes = torch.stack([
        new_x, new_y, z, new_w, new_l, h, new_cos_theta, new_sin_theta
    ], dim=-1)
    
    norm_boxes = norm_boxes.reshape(boxes.shape) if is_batch else norm_boxes

    return norm_boxes

def timestep_embedding(timesteps, dim, type_target, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(type_target.dtype)

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        dropout,
        out_channels=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = in_channels if out_channels is None else out_channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, self.in_channels),
            nn.SiLU(),
            nn.Conv1d(self.in_channels, self.out_channels, 1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.emb_channels, self.out_channels)
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(self.out_channels, self.out_channels, 1)
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(self.in_channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        h = h + emb_out.unsqueeze(-1)
        h = self.out_layers(h)
        return self.skip_connection(x) + h

def exists(val):
    return val is not None

def uniq(arr):
    return{el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

def getColliPtsRadius() -> torch.Tensor:
    # returns a list of 13 elements, containing radius of the 13 collision points
    return [MOMA_PARAM.chassis_colli_radius] + [r for r in MOMA_PARAM.colli_point_radius if r >= 1e-3]

def getColliMatrix() -> torch.Tensor:
    # A 12 x 12 matrix, indicating the collision between two collision points
    # True indicates potential collision, False indicates no collision
    return torch.tensor(MOMA_PARAM.collision_matrix) < 0

def getColliPts(state_batch : torch.Tensor) -> torch.Tensor:
    """autograd supported function to get collision points from state

    Args:
        state (torch.Tensor): N x 11 tensor containing state information, where N is the batch size, 11 is the state dimension

    Returns:
        torch.Tensor: N x 13 x 3 tensor containing collision points, where N is the batch size, 13 is the number of collision points, 3 is the dimension of the point
    """    
    _size = state_batch.size(0)
    device = state_batch.device
    dtype = state_batch.dtype

    dof_num = 7

    # RO tensor
    R0_tensor = torch.zeros((_size, 3, 3), dtype=dtype, device=device)
    R0_tensor[:, 0, 0] = state_batch[:, 2]
    R0_tensor[:, 0, 1] = -state_batch[:, 3]
    R0_tensor[:, 1, 0] = state_batch[:, 3]
    R0_tensor[:, 1, 1] = state_batch[:, 2]
    R0_tensor[:, 2, 2] = 1

    p0_tensor = torch.zeros((_size, 3, 1), dtype=dtype, device=device)
    p0_tensor[:, :2, 0] = state_batch[:, :2]
    p0_tensor[:, 2, 0] = robot_param.chassis_height

    p0_tensor = p0_tensor.clone()
    p0_tensor =  p0_tensor + R0_tensor @ torch.tensor(robot_param.relative_t, dtype=dtype, device=device).reshape((-1,1))
    
    R0_tensor = R0_tensor.clone()
    R0_tensor =  R0_tensor @ torch.tensor(robot_param.relative_R, dtype=dtype, device=device) # broad casting

    # R_tensor = torch.zeros((dof_num + 1, _size, 3, 3))
    # R_tensor[0] = R0_tensor
    R_tensor = [R0_tensor]

    # p_tensor = torch.zeros((dof_num + 1, _size, 3))
    p_tensor = [p0_tensor.squeeze(2)] 

    for dof in range(dof_num):
        cos_theta = torch.cos(state_batch[:, 4+dof])
        sin_theta = torch.sin(state_batch[:, 4+dof])
        dof_R = torch.zeros((_size, 3, 3), dtype=state_batch.dtype, device=device)
        if (dof % 2) == 0:
            dof_R[:, 0, 0] = cos_theta
            dof_R[:, 0, 1] = -sin_theta
            dof_R[:, 1, 0] = sin_theta
            dof_R[:, 1, 1] = cos_theta
            dof_R[:, 2, 2] = 1
        else:
            dof_R[:, 0, 0] = cos_theta
            dof_R[:, 0, 2] = sin_theta
            dof_R[:, 1, 1] = 1
            dof_R[:, 2, 0] = -sin_theta
            dof_R[:, 2, 2] = cos_theta
        # p_tensor = p_tensor.clone()
        # p_tensor[dof+1] = p_tensor[-1] + R_tensor[dof, :, :, 2] * robot_param.colli_length[dof]
        # R_tensor = R_tensor.clone()
        # R_tensor[dof+1] = R_tensor[dof] @ dof_R
        p_tensor.append(p_tensor[-1] + R_tensor[-1][:,:,2] * robot_param.colli_length[dof])
        R_tensor.append(R_tensor[-1] @ dof_R)

    colli_pts = torch.zeros ((_size, 13, 3), dtype=dtype, device=device)
    colli_pts[:, 0, :2] = state_batch[:, :2] # chassis colli point

    cnum = 1
    for i in range(dof_num+1):
        for j in range(2):
            if(robot_param.colli_points[i*2+j] == 0):
                continue
            # colli_pts[:, cnum, :] = (p_tensor[i, :] + R_tensor[i, :, :, 2] * robot_param.colli_points[i*2+j])
            colli_pts[:, cnum, :] = (p_tensor[i] + R_tensor[i][:, :, 2] * robot_param.colli_points[i*2+j])

            cnum += 1
    return colli_pts

def getTrajPts(state_batch : torch.Tensor) -> torch.Tensor:
    """autograd supported function to get collision points from state

    Args:
        state (torch.Tensor): N x 11 tensor containing state information, where N is the batch size, 11 is the state dimension

    Returns:
        torch.Tensor: N x 13 x 4 tensor containing collision points, where N is the batch size, 13 is the number of collision points, 3 is the dimension of the point
    """    
    _size = state_batch.size(0)
    device = state_batch.device
    dtype = state_batch.dtype

    dof_num = 7

    # RO tensor
    R0_tensor = torch.zeros((_size, 3, 3), dtype=dtype, device=device)
    R0_tensor[:, 0, 0] = state_batch[:, 2]
    R0_tensor[:, 0, 1] = -state_batch[:, 3]
    R0_tensor[:, 1, 0] = state_batch[:, 3]
    R0_tensor[:, 1, 1] = state_batch[:, 2]
    R0_tensor[:, 2, 2] = 1

    p0_tensor = torch.zeros((_size, 3, 1), dtype=dtype, device=device)
    p0_tensor[:, :2, 0] = state_batch[:, :2]
    p0_tensor[:, 2, 0] = robot_param.chassis_height

    p0_tensor = p0_tensor.clone()
    p0_tensor =  p0_tensor + R0_tensor @ torch.tensor(robot_param.relative_t, dtype=dtype, device=device).reshape((-1,1))
    
    R0_tensor = R0_tensor.clone()
    R0_tensor =  R0_tensor @ torch.tensor(robot_param.relative_R, dtype=dtype, device=device) # broad casting

    # R_tensor = torch.zeros((dof_num + 1, _size, 3, 3))
    # R_tensor[0] = R0_tensor
    R_tensor = [R0_tensor]

    # p_tensor = torch.zeros((dof_num + 1, _size, 3))
    p_tensor = [p0_tensor.squeeze(2)] 

    for dof in range(dof_num):
        cos_theta = torch.cos(state_batch[:, 4+dof])
        sin_theta = torch.sin(state_batch[:, 4+dof])
        dof_R = torch.zeros((_size, 3, 3), dtype=state_batch.dtype, device=device)
        if (dof % 2) == 0:
            dof_R[:, 0, 0] = cos_theta
            dof_R[:, 0, 1] = -sin_theta
            dof_R[:, 1, 0] = sin_theta
            dof_R[:, 1, 1] = cos_theta
            dof_R[:, 2, 2] = 1
        else:
            dof_R[:, 0, 0] = cos_theta
            dof_R[:, 0, 2] = sin_theta
            dof_R[:, 1, 1] = 1
            dof_R[:, 2, 0] = -sin_theta
            dof_R[:, 2, 2] = cos_theta
        # p_tensor = p_tensor.clone()
        # p_tensor[dof+1] = p_tensor[-1] + R_tensor[dof, :, :, 2] * robot_param.colli_length[dof]
        # R_tensor = R_tensor.clone()
        # R_tensor[dof+1] = R_tensor[dof] @ dof_R
        p_tensor.append(p_tensor[-1] + R_tensor[-1][:,:,2] * robot_param.colli_length[dof])
        R_tensor.append(R_tensor[-1] @ dof_R)

    colli_pts = torch.zeros ((_size, 13, 4), dtype=dtype, device=device)
    colli_pts[:, 0, :2] = state_batch[:, :2] # chassis colli point
    
    colli_radius = torch.tensor(getColliPtsRadius(), dtype=dtype, device=device)
    colli_pts[:, :, 3] = colli_radius.unsqueeze(0).repeat(state_batch.size(0), 1)
    cnum = 1
    for i in range(dof_num+1):
        for j in range(2):
            if(robot_param.colli_points[i*2+j] == 0):
                continue
            # colli_pts[:, cnum, :] = (p_tensor[i, :] + R_tensor[i, :, :, 2] * robot_param.colli_points[i*2+j])
            colli_pts[:, cnum, :3] = (p_tensor[i] + R_tensor[i][:, :, 2] * robot_param.colli_points[i*2+j])

            cnum += 1
    return colli_pts

class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, F, N)."""
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class GeoCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.inner_dim = inner_dim
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask_q=None, mask_kv=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        # print(q.shape, k.shape)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        #! Note: mask_q shape: (B, Nq); mask_kv shape: (B, Nk)
        if exists(mask_q):
            max_neg_value = -torch.finfo(sim.dtype).max
            klen = k.shape[-2]
            mask_q = repeat(mask_q, 'b j -> (b h) j k', h=h, k=klen)
            sim.masked_fill_(mask_q, max_neg_value)
        if exists(mask_kv):
            max_neg_value = -torch.finfo(sim.dtype).max
            qlen = q.shape[-2]
            mask_kv = repeat(mask_kv, 'b j -> (b h) k j', h=h, k=qlen)
            sim.masked_fill_(mask_kv, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.inner_dim = inner_dim
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask_q=None, mask_kv=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        if True:
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            #! Note: mask_q shape: (B, Nq); mask_kv shape: (B, Nk)
            if exists(mask_q):
                max_neg_value = -torch.finfo(sim.dtype).max
                klen = k.shape[-2]
                mask_q = repeat(mask_q, 'b j -> (b h) j k', h=h, k=klen)
                sim.masked_fill_(mask_q, max_neg_value)
            if exists(mask_kv):
                max_neg_value = -torch.finfo(sim.dtype).max
                qlen = q.shape[-2]
                mask_kv = repeat(mask_kv, 'b j -> (b h) k j', h=h, k=qlen)
                sim.masked_fill_(mask_kv, max_neg_value)

            attn = sim.softmax(dim=-1)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)
        if flash_attn is None:
            # Do w/o flash_attn acceleration
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            #! Note: mask_q shape: (B, Nq); mask_kv shape: (B, Nk)
            if exists(mask_q):
                max_neg_value = -torch.finfo(sim.dtype).max
                klen = k.shape[-2]
                mask_q = repeat(mask_q, 'b j -> (b h) j k', h=h, k=klen)
                sim.masked_fill_(mask_q, max_neg_value)
            if exists(mask_kv):
                max_neg_value = -torch.finfo(sim.dtype).max
                qlen = q.shape[-2]
                mask_kv = repeat(mask_kv, 'b j -> (b h) k j', h=h, k=qlen)
                sim.masked_fill_(mask_kv, max_neg_value)

            attn = sim.softmax(dim=-1)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)
        else:
            #! Yet to be tested
            # Accelerate w/ flash_attn
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h).to(torch.float16), (q, k, v))
            _b, _n, _h, _d = q.shape
            _bv,_nv,_hv,_dv = v.shape
            #HACK vanilla flash_attn cannot handle attention mask, need to hack through
            out = None
            if exists(mask_q):      # w/ q_mask
                k = torch.cat([k, torch.ones((_bv, _nv, _hv, 1)).to(k.device)], dim=-1).half()
                v = torch.cat([v, torch.zeros((_bv, _nv, _hv, 1)).to(v.device)], dim=-1).half()
                _add = torch.zeros(mask_q.shape, dtype=torch.float16, device=mask_q.device)        # B,N
                _add.masked_fill_(mask_q, -torch.inf)   # B,N
                q = torch.cat([q, repeat(_add, 'b n -> b n h 1', h=h).to(q.device)], dim=-1).half()

                out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, \
                    softmax_scale=self.scale, causal=False)
                out = out[:, :, :, :-1]
                
            elif exists(mask_kv):   # w/ kv_mask
                q = torch.cat([q, torch.ones((_b, _n, _h, 1)).to(q.device)], dim=-1).half()
                _add = torch.zeros(mask_kv.shape, dtype=torch.float16, device=mask_kv.device)       # B,N
                _add.masked_fill_(mask_kv, -torch.inf)  # B,N
                k = torch.cat([k, repeat(_add, 'b n -> b n h 1', h=h).to(k.device)], dim=-1).half()
                v = torch.cat([v, torch.zeros((_bv, _nv, _hv, 1)).to(v.device)], dim=-1).half()
                out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, \
                    softmax_scale=self.scale, causal=False)
                out = out[:, :, :, :-1]
            else:
                # w/o mask
                out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, \
                    softmax_scale=self.scale, causal=False)
            
            out = rearrange(out, 'b n h d -> b n (h d)', h=h).half()
            # print("inner dim: ", self.inner_dim)
            # print("q shape: ", q.shape)
            # print("k shape: ", k.shape)
            # print(out.shape)
            return self.to_out(out)
            
    
class SGOccFusion(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., gated_ff=True, mult_ff=2):
        super().__init__()
        self.sa_sg = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.sa_occ = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        
        self.ff_sg = FeedForward(dim, dropout=dropout, glu=gated_ff, mult=mult_ff)
        self.ff_occ = FeedForward(dim, dropout=dropout, glu=gated_ff, mult=mult_ff)
        
        self.ca_sg = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.ca_occ = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none

        self.norm1sg = nn.LayerNorm(dim)
        self.norm2sg = nn.LayerNorm(dim)
        self.norm3sg = nn.LayerNorm(dim)
        self.norm1occ = nn.LayerNorm(dim)
        self.norm2occ = nn.LayerNorm(dim)
        self.norm3occ = nn.LayerNorm(dim)

    def forward(self, sg, occ, occ_mask=None):
        sgx_a = self.sa_sg(self.norm1sg(sg), context=None) + sg
        occx_a = self.sa_occ(self.norm1occ(occ), context=None, mask_q=occ_mask) + occ

        sgx = self.ca_sg(self.norm2sg(sgx_a), context=occx_a, mask_kv=occ_mask) + sgx_a
        sgx = self.ff_sg(self.norm3sg(sgx)) + sgx

        occx = self.ca_occ(self.norm2occ(occx_a), context=sgx_a, mask_q=occ_mask) + occx_a
        occx = self.ff_occ(self.norm3occ(occx)) + occx

        return torch.cat([occx, sgx], dim=1)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, mult_ff=2):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, mult=mult_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None, mask=None):
        x = self.attn1(self.norm1(x), context=None, mask_q=mask) + x
        x = self.attn2(self.norm2(x), context=context, mask_q=mask) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    """
    Transformer block for sequential data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to sequential data.
    """
    def __init__(self, in_channels, n_heads=8, d_head=64,
                 depth=1, dropout=0., context_dim=None, mult_ff=2):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv1d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, mult_ff=mult_ff)
                for d in range(depth)]
        )

        self.proj_out = nn.Conv1d(inner_dim,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        B, C, L,  = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)

        x = rearrange(x, 'b c l -> b l c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b l c -> b c l')
        x = self.proj_out(x)
        return x + x_in

if __name__ == '__main__':
    st = SpatialTransformer(256, 8, 64, 6, context_dim=768)
    a = torch.rand(2, 256, 10)
    context = torch.rand(2, 5, 768)
    o = st(a, context=context)
    # print(o.shape)

    torch.autograd.set_detect_anomaly(True)

    with torch.autograd.detect_anomaly():
        test = torch.ones((256, 11), dtype=torch.float32)
        test[:, 3] = 0
        test.requires_grad = True
        colli_pts = getColliPts(test)
        loss = torch.sum(colli_pts[0])
        loss.backward()
        # print(test.grad[0])
    
    # test fk
    a = torch.rand(1, 10)
    b = a.transpose(0, 1).numpy()
    a.requires_grad_()
    
    ap = torch.zeros(1, 11)
    ap[:, :2] = a[:, :2]
    ap[:, 2] = torch.cos(a[:, 2])
    ap[:, 3] = torch.sin(a[:, 2])
    ap[:, 4:] = a[:, 3:]
    
    cpp = robot_param.getFKPose(b)
    pyth = get_fk_pose(ap)
    print("kinematics error:", (pyth[0]-torch.as_tensor(cpp)).max().item())
    
    g = torch.rand(9, 1).numpy()
    cpp = robot_param.getEEGrads(b, g)
    s = (pyth * torch.as_tensor(g).transpose(0, 1)).sum()
    s.backward()
    print("gradient error:", (a.grad[0]- torch.as_tensor(cpp)).max().item())
        
    