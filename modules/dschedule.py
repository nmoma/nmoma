from typing import Dict, List, Tuple

import torch
import math

def exponential_beta_schedule(n_diffusion_steps, beta_start=1e-4, beta_end=1.0):
    # exponential increasing noise from t=0 to t=T
    beta_start = torch.as_tensor(beta_start, dtype=torch.float32)
    beta_end = torch.as_tensor(beta_end, dtype=torch.float32)
    x = torch.linspace(0, n_diffusion_steps, n_diffusion_steps)
    a = 1 / n_diffusion_steps * torch.log(beta_end / beta_start)
    return beta_start * torch.exp(a * x)

def make_schedule_ddpm(timesteps: int, beta: List, beta_schedule: str, s=0.008) -> Dict:
    assert beta[0] < beta[1] < 1.0
    if beta_schedule == 'linear':
        betas = torch.linspace(beta[0], beta[1], timesteps)
    elif beta_schedule == 'cosine':
        x = torch.linspace(0, timesteps, timesteps + 1, dtype = torch.float32)
        #! xulong float64
        # x = torch.linspace(0, timesteps, timesteps + 1, dtype = torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999)
    elif beta_schedule == 'sqrt':
        betas = torch.sqrt(torch.linspace(beta[0], beta[1], timesteps))
    elif beta_schedule == 'exp':
        x = torch.linspace(0, timesteps, timesteps)
        beta_start = torch.as_tensor(beta[0], dtype=torch.float32)
        beta_end = torch.as_tensor(beta[1], dtype=torch.float32)
        a = 1 / timesteps * torch.log(beta_end / beta_start)
        betas = beta_start * torch.exp(a * x)
    else:
        raise Exception('Unsupport beta schedule.')

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])    
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return {
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1 - alphas_cumprod),
        'log_one_minus_alphas_cumprod': torch.log(1 - alphas_cumprod),
        'sqrt_recip_alphas_cumprod': torch.sqrt(1 / alphas_cumprod),
        'sqrt_recipm1_alphas_cumprod': torch.sqrt(1 / alphas_cumprod - 1),
        'posterior_variance': posterior_variance,
        'posterior_log_variance_clipped': torch.log(posterior_variance.clamp(min=1e-20)),
        'posterior_mean_coef1': betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod),
        'posterior_mean_coef2': (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
    }

if __name__ == '__main__':
    make_schedule_ddpm(10, [1.0e-4, 0.01], 'linear', **{'s': 0.008})
    make_schedule_ddpm(10, [1.0e-4, 0.01], 'cosine', **{'s': 0.008})
    make_schedule_ddpm(10, [1.0e-4, 0.01], 'sqrt', **{'s': 0.008})
    make_schedule_ddpm(10, [1.0e-4, 0.01], 'exp', **{'s': 0.008})
    