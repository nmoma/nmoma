import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Functional
from omegaconf import DictConfig
from utils.data_loader import MomaTrajTrans
from loss import getDistanceFromState, positiveSmoothedL1, positiveSmoothedL2, positiveSmoothedL3, positiveSmoothedExp
from utils.utils import colli_point_radius

class CostGuide(nn.Module):
    def __init__(self, transer:MomaTrajTrans, 
                 cfg: DictConfig):
        super().__init__()
        self.transer = transer
        self.cost_guides = [SDFGuide, SmoothGuide]
        self.clip_grad = cfg.clip_grad
        self.max_grad_norm = cfg.max_grad_norm
        self.weights = cfg.weights

    def clip_grad_by_norm(self, grad):
        # clip gradient by norm
        if self.clip_grad:
            grad_norm = torch.linalg.norm(grad + 1e-6, dim=-1, keepdims=True)
            scale_ratio = torch.clip(grad_norm, 0.0, self.max_grad_norm) / grad_norm
            grad = scale_ratio * grad
        return grad
    
    def gradients(self, trajin: torch.Tensor, data: dict):
        traj = trajin.clone().detach()
        with torch.enable_grad():
            if not traj.requires_grad:
                traj.requires_grad_()
            de_traj = self.transer.detrans((traj, data["startgoal"][:, 7:10].detach()))
            cost = 0.0
            for i in range(len(self.cost_guides)):
                cost += self.weights[i] * self.cost_guides[i](de_traj, data)
            print("cost:", cost.item())
            cost.backward()
            grad = traj.grad.clone()
            # grad = torch.zeros_like(traj)
            # print("grad:", grad.norm().item())
            grad = self.clip_grad_by_norm(grad)
        return grad
      
def SDFGuide(traj, data, margin : float = 0.0):
    _, W, _ = traj.shape
    sdf = torch.repeat_interleave(data['sdf'].detach().flatten(start_dim=1), W, dim=0)
    dist =  - getDistanceFromState(traj.reshape(-1, 11), sdf) + colli_point_radius.to(traj.device) + margin
    # cost = dist[dist>0].flatten().sum()
    cost = positiveSmoothedL2(dist.flatten()).mean()
    return cost

def SmoothGuide(traj, data):
    vel = traj[:, 1:, :] - traj[:, :-1, :]
    acc = vel[:, 1:, :] - vel[:, :-1, :]
    jerk = acc[:, 1:, :] - acc[:, :-1, :]
    return (jerk**2).sum()

def SafeCost(traj, data, redution = "mean", margin : float = 0.0):
    B, W, S = traj.shape
    sdf = torch.repeat_interleave(data['sdf'].to(traj.device).detach().flatten(start_dim=1), W, dim=0)
    dist =  - getDistanceFromState(traj.reshape(-1, S), sdf) + colli_point_radius.to(traj.device) + margin
    cost = positiveSmoothedL2(dist).reshape(B, W, dist.shape[-1]).mean(dim=2).mean(dim=1)
    if redution == "mean":
        cost = cost.mean()
    return cost

def SteepSafeCost(traj, data, redution = "mean", margin : float = 0.0):
    B, W, S = traj.shape
    sdf = torch.repeat_interleave(data['sdf'].to(traj.device).detach().flatten(start_dim=1), W, dim=0)
    dist =  - getDistanceFromState(traj.reshape(-1, S), sdf) + colli_point_radius.to(traj.device) + margin
    cost = positiveSmoothedExp(dist).reshape(B, W, dist.shape[-1]).mean(dim=2).mean(dim=1)
    if redution == "mean":
        cost = cost.mean()
    return cost

def SafeRate(traj, data, margin : float = 0.0):
    _, W, _ = traj.shape
    sdf = torch.repeat_interleave(data['sdf'].to(traj.device).detach().flatten(start_dim=1), W, dim=0)
    dist =  - getDistanceFromState(traj.reshape(-1, 11), sdf) + colli_point_radius.to(traj.device) + margin
    colli_num = (dist > 0).any(dim=1).sum()
    rate = 1.0 - colli_num / dist.shape[0]
    return rate

def VelCost(traj, data, redution = "mean"):
    vel = traj[:, 1:, :] - traj[:, :-1, :]
    loss = torch.norm(vel, dim=2).mean(dim=1)
    if redution == "mean":
        loss = loss.mean()
    return loss

def AccCost(traj, data):
    zeros_tensor = torch.zeros_like(traj[:, 1:2, :])
    traj1 = torch.cat([zeros_tensor, traj[:, 1:, :]], dim=1)
    traj2 = torch.cat([traj[:, :-1, :], zeros_tensor], dim=1)
    vel = traj1 - traj2
    zeros_tensor = torch.zeros_like(vel[:, 1:2, :])
    vel1 = torch.cat([zeros_tensor, vel[:, 1:, :]], dim=1)
    vel2 = torch.cat([vel[:, :-1, :], zeros_tensor], dim=1)
    return torch.mean(torch.norm(vel1 - vel2, dim=2))

def JerkCost(traj, data):
    zeros_tensor = torch.zeros_like(traj[:, 1:2, :])
    traj1 = torch.cat([zeros_tensor, traj[:, 1:, :]], dim=1)
    traj2 = torch.cat([traj[:, :-1, :], zeros_tensor], dim=1)
    vel = traj1 - traj2
    zeros_tensor = torch.zeros_like(vel[:, 1:2, :])
    vel1 = torch.cat([zeros_tensor, vel[:, 1:, :]], dim=1)
    vel2 = torch.cat([vel[:, :-1, :], zeros_tensor], dim=1)
    acc = vel1 - vel2
    zeros_tensor = torch.zeros_like(acc[:, 1:2, :])
    acc1 = torch.cat([zeros_tensor, acc[:, 1:, :]], dim=1)
    acc2 = torch.cat([acc[:, :-1, :], zeros_tensor], dim=1)
    return torch.mean(torch.norm(acc1 - acc2, dim=2))

if __name__ == "__main__":
    a = torch.rand(4, 16, 11, requires_grad=True)
    # p = SmoothGuide(a, None)
    p = SDFGuide(a, None)
    p.backward()
    print(a)
    print(a.grad)
    