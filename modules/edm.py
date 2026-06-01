import os
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm

from utils.data_loader import MomaTrajTrans
from modules.guidance import *
from loss import NormedSmoothLoss, UniArcLoss, FocalLoss

def identity(t, *args, **kwargs):
    return t

# P_mean=-1.2, P_std=1.2, sigma_data=0.5
class EDM(torch.nn.Module):
    def __init__(self, transer: MomaTrajTrans, 
                 eps_model: nn.Module, cfg: DictConfig) -> None:
        super(EDM, self).__init__()
        
        self.transer = transer
        self.model = eps_model
        self.num_steps = cfg.steps
        self.loss_weight = torch.zeros(len(cfg.loss_weight))

        self.sigma_min = cfg.sigma_min
        self.sigma_max = cfg.sigma_max
        self.sigma_data = cfg.sigma_data
        self.P_mean = cfg.P_mean
        self.P_std = cfg.P_std

        trajlib_length = np.load('logs/trajlib_length.npy', allow_pickle=True)
        alpha = torch.tensor(trajlib_length).float().cuda()
        self.criterion_class = FocalLoss(alpha=alpha, gamma=2)
        
        if cfg.load_pretrained:
            self.load_ckpt(cfg.pretrained_path)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def getDx(self, x, sigma, condition):
        sigma = sigma.reshape(-1, 1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(c_in * x, c_noise.flatten(), condition)
        D_x = c_skip * x + c_out * F_x
        return D_x
    
    def forward(self, data: Dict):
        x = data["x"]

        rnd_normal = torch.randn([x.shape[0], 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x) * sigma

        condtion = self.model.condition(data)
        ret = dict()
        if self.loss_weight[0] > 0:
            output = self.getDx(x + n, sigma, condtion)
            rec_loss = weight * ((output - x) ** 2)
            ret['rec'] = rec_loss.mean() * self.loss_weight[0]
            de_traj = self.transer.detrans((output, data["startgoal"][:, 7:10].detach()))
            de_gt = self.transer.detrans((data['x'], data["startgoal"][:, 7:10])).detach()
            ret['sdf'] = SafeCost(de_traj, data) * self.loss_weight[1] if self.loss_weight[1] > 0 else torch.as_tensor(0.0).to(de_traj.device)
            ret['smooth'] = NormedSmoothLoss(de_traj, de_gt) * self.loss_weight[2] if self.loss_weight[2] > 0 else torch.as_tensor(0.0).to(de_traj.device)
            ret['arc'] = UniArcLoss(de_traj, de_gt) * self.loss_weight[3] if self.loss_weight[3] > 0 else torch.as_tensor(0.0).to(de_traj.device)

        if self.loss_weight[-1] > 0:
            cls_output = self.model.classify(condtion)
            ret['classify'] = self.criterion_class(cls_output, data['min_idx'].view(-1)) * self.loss_weight[-1]

        loss = 0.0
        for key in ret:
            loss += ret[key]
            
        ret['loss'] = loss
        return ret
    
    @torch.no_grad()
    def sample(self, trajlib: torch.Tensor, data: Dict) -> torch.Tensor:
        condition = self.model.condition(data)
        prob = self.model.classify(condition)
        prob = torch.softmax(prob, dim=1)
        prbs, pred_idxs = torch.topk(prob[0], prob.shape[0])
        
        predict_prims = trajlib[pred_idxs.long().cpu()].to(condition.device)
        predict_prims = (predict_prims - data['mu']) / data['std']
        
        #TODO: manage params
        sigma_min=0.002
        sigma_max=80
        rho=7
        S_churn=0
        S_min=0
        S_max=float('inf')
        S_noise=1

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.sigma_min)
        sigma_max = min(sigma_max, self.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=condition.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (self.num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.

        noise = torch.randn_like(data['x'], device=condition.device)
        x_next = noise.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / self.num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = self.getDx(x_hat.float(), t_hat.float(), condition).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                denoised = self.getDx(x_next.float(), t_next.float(), condition).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next.unsqueeze(1).unsqueeze(1).float(), predict_prims, prbs
    
    def load_ckpt(self, path: str) -> None:
        assert os.path.exists(path), 'Can\'t find provided ckpt.'

        saved_state_dict = torch.load(path)['model']
        model_state_dict = self.state_dict()

        for key in model_state_dict:
            if key in saved_state_dict:
                model_state_dict[key] = saved_state_dict[key]
            if 'module.'+key in saved_state_dict:
                model_state_dict[key] = saved_state_dict['module.'+key]
        self.load_state_dict(model_state_dict)
    