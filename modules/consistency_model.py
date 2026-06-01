# This file contains the implementation of the consistency model.
# It performs consistency distillation on the pre-trained DDPM model.
# see https://arxiv.org/abs/2303.01469 for more details.

#? Is it better to sample startgoal directly, or use the a dataset? Probably the former.
#? Probably better to implement the free trick implemented in the paper.
#TODO Data transformation mess, probably no need

import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_loader import MomaDataset, collate_fn_4ptrans
from unet import UNetModel
from ddpm import DDPM
from dschedule import make_schedule_ddpm

import numpy as np
from omegaconf import OmegaConf, DictConfig
from einops import rearrange


#TODO move hyperparameters to config file
__BATCH_SIZE__ = 16
__EPOCHS__ =100
__EMA_DECAY__ = 0.999

class ConsistencyModel(nn.Module):
    def __init__(self):
        super(ConsistencyModel, self).__init__()

        self.fc1 = nn.Linear(11*16+1, 256)
        self.act1= nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.act2= nn.ReLU()
        self.fc3 = nn.Linear(256, 11*16)

    def forward(self, x_t, t):
        """Predict x_0 from x_t at step t

        Args:
            x_t (torch.Tensor): trajectory at step t
            t (int): step number associated with x_t
        """        
        # TODO use unet here
        batch_size = x_t.shape[0]
        x = torch.cat([x_t.flatten(start_dim=1), torch.full((batch_size,1), fill_value=t)], dim=1)
        x1 = self.act1(self.fc1(x))
        x2 = self.act2(self.fc2(x1))
        x3 = self.fc3(x2)
        out= rearrange(x3, 'b (h w) -> b h w', h=16, w=11)
        return out

def forward_diffusion(data : torch.Tensor, t : int, ddpm_model : torch.nn.Module):
    """Implements forward process of the diffussion model.

    Args:
        data (Dict): data['x'] with B * 16 * 11
        t (int): tensor of shape (B) of Long type
        beta (torch.Tensor): tensor of shape (T) of Float type, where T is the number of diffusion steps, which is model-specific.

    Returns:
        Tuple(torch.Tensor, torch.Tensor): (noisy_x, noise), data['x'] with added noise complying to variance schedule of the ddpm_model.
    """        
    # TODO check this function, does not seem correct
    x = data['x']
    batch_size = x.shape[0]
    _t = torch.full((batch_size,), fill_value=t)
    noise = torch.randn_like(x)
    noisy_x = ddpm_model.sqrt_alphas_cumprod[t].view(-1, 1, 1) * x + (1 - ddpm_model.alphas_cumprod)[t].view(-1, 1, 1) * noise

    return noisy_x, noise

def train(online_model, ddpm_model, dataloader):
    # see algorithm 2 in https://arxiv.org/abs/2303.01469
    target_model = copy.deepcopy(online_model)
    target_model.eval() # used to compute loss, updated using EMA
    online_model.train()
    optimizer = optim.Adam(online_model.parameters(), lr=1e-3)
    beta = ddpm_model.betas
    # This function update the target model using EMA
    def _weighting(ts):
        return 1.0 / ddpm_model.posterior_variance[ts]

    def _update_ema():
        with torch.no_grad():
            for ema_param, param in zip(target_model.parameters(), online_model.parameters()):
                ema_param.data = __EMA_DECAY__ * ema_param.data + (1 - __EMA_DECAY__) * param.data

    def _make_data(data, device=None):
        # returns a suitable data dictionary for the DDPM model
        data['pos'] = data['pos'].repeat((1,1,1))
        offset, count = [], 0
        for item in data['pos']:
            count += item.shape[1]
            offset.append(count)
        offset = torch.IntTensor(offset)
        data['offset'] = offset
        data['pos'] = rearrange(data['pos'], 'b n c -> (b n) c')
        return data

    for epoch in range(__EPOCHS__):
        for batch_idx, data in enumerate(dataloader):
            # data = {k: v.cuda() for k, v in data.items()}

            # forward diffusion random number of steps, t, to sample x2
            data = _make_data(data)
            # t = torch.randint(0, len(beta), (data['x'].shape[0],))
            t = torch.randint(0, len(beta), (1,)).item()
            noisy_x, _ = forward_diffusion(data, t, ddpm_model)
            x2 = noisy_x.detach().clone()
            x1 = None
            
            # one step backward from x2 to get x1, with step_num=(t-1)
            with torch.no_grad():
                x1 = ddpm_model.apply_observation(x2, data)
                condition = ddpm_model.eps_model.condition(data) #TODO check if this is correct
                data['cond'] = condition
                x1 = ddpm_model.p_sample(x2, t, data)
                # predict with the target model
                target_pred = target_model(x1, t-1)

            # sample from the ddpm model
            online_pred = online_model(x2, t)

            weights = _weighting(t-1)
            loss = weights * F.mse_loss(target_pred, online_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _update_ema()
        print(f"Epoch {epoch+1}/{__EPOCHS__}, Loss: {loss.item()}")
            
def main():
    cfg = OmegaConf.load("config/ddpm.yaml")

    trainset = MomaDataset("data_valid_raw.h5", cfg.data_normer)
    dataloader = DataLoader(
        trainset,
        batch_size=__BATCH_SIZE__,
        collate_fn=collate_fn_4ptrans,
        pin_memory=True,
        drop_last = True,
        shuffle=True
    )

    unet = UNetModel(cfg, slurm=False)
    ddpm_model = DDPM(trainset.transer, unet, cfg.diffuser)

    # betas = model.betas
    # print(betas.shape)

    online_model = ConsistencyModel()

    train(online_model, ddpm_model, dataloader)

if __name__ == '__main__':
    main()
    