import os
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from omegaconf import DictConfig
from tqdm import tqdm

from modules.dschedule import make_schedule_ddpm
from utils.data_loader import MomaTrajTrans
from modules.guidance import *
from loss import NormedSmoothLoss, UniArcLoss, FocalLoss, UniEEArcLoss, AnchorLoss

def identity(t, *args, **kwargs):
    return t

class DDPM(nn.Module):
    def __init__(self, transer: MomaTrajTrans, 
                 eps_model: nn.Module, cfg: DictConfig) -> None:
        super(DDPM, self).__init__()
        self.transer = transer
        self.eps_model = eps_model
        self.timesteps = cfg.steps
        self.schedule_cfg = cfg.schedule_cfg
        self.rand_t_type = cfg.rand_t_type

        self.has_observation = cfg.has_obs

        self.ddim_use = cfg.ddim_use
        self.ddimsteps = cfg.ddim_steps
        self.ddim_eta = cfg.ddim_eta
        
        self.trunc_use = cfg.trunc_use
        self.trunc_type = cfg.trunc_type
        self.trunc_steps = cfg.trunc_steps
        
        self.objective = cfg.objective
        self.class_loss_type = cfg.class_loss_type
        
        # guidance
        self.add_guidance = cfg.add_guidance
        self.guidin_time = cfg.guidin_time
        self.guidin_num_per_time = cfg.guidin_num_per_time
        self.guidance = CostGuide(transer, cfg.guidance)

        for k, v in make_schedule_ddpm(self.timesteps, **self.schedule_cfg).items():
            self.register_buffer(k, v)
        
        if cfg.loss_type == 'l1':
            self.criterion = F.l1_loss
        elif cfg.loss_type == 'l2':
            self.criterion = F.mse_loss
        else:
            raise Exception('Unsupported loss type.')
        
        if cfg.class_loss_type == 'cross_entropy':
            self.criterion_class = nn.CrossEntropyLoss()
        elif cfg.class_loss_type == 'weighted_cross_entropy':
            idx_counts = np.load('logs/class_counts.npy', allow_pickle=True)
            class_weights = 1.0 / idx_counts
            class_weights = class_weights / class_weights.sum()
            self.criterion_class = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().to(self.device))
        elif cfg.class_loss_type == 'focal_loss':
            if self.trunc_use and self.trunc_type == "anchor":
                self.criterion_class = AnchorLoss()
            else:
                trajlib_length = np.load('logs/trajlib_length.npy', allow_pickle=True)
                alpha = torch.tensor(trajlib_length).float().to(self.device)
                self.criterion_class = FocalLoss(alpha=alpha, gamma=2)
        elif cfg.class_loss_type == 'kl_div':
            # self.criterion_class = nn.CrossEntropyLoss()
            pass
        else:
            raise Exception('Unsupported class loss type.')

        print("initialize trajlib32")
        self.trajlib = np.load('logs/trajlib32.npy', allow_pickle=True)
        self.trajlib = torch.as_tensor(self.trajlib).float().contiguous()
        
        self.loss_weight = torch.zeros(len(cfg.loss_weight))
        
        if cfg.load_pretrained:
            self.load_ckpt(cfg.pretrained_path)

    @property
    def device(self):
        return self.betas.device
    
    def apply_observation(self, x_t: torch.Tensor, data: Dict) -> torch.Tensor:
        """ Apply observation to x_t, if self.has_observation if False, this method will return the input

        Args:
            x_t: noisy x in step t
            data: original data provided by dataloader
        """
        ## has start observation, used in path planning and start-conditioned motion generation
        if self.has_observation and 'startgoal' in data:
            start = data['x'][:, 0].clone().detach()
            goal = data['x'][:, -1].clone().detach()

            if start.shape[0] != x_t.shape[0]:
                libnum = x_t.shape[0] // start.shape[0]
                start = start.repeat(libnum, 1)
                goal = goal.repeat(libnum, 1)
            
            x_t[:, 0, :] = start
            x_t[:, -1, :] = goal
        
        return x_t
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """ Forward difussion process, $q(x_t \mid x_0)$, this process is determinative 
        and has no learnable parameters.

        $x_t = \sqrt{\bar{\alpha}_t} * x0 + \sqrt{1 - \bar{\alpha}_t} * \epsilon$

        Args:
            x0: samples at step 0
            t: diffusion step
            noise: Gaussian noise
        
        Return:
            Diffused samples
        """
        B, *x_shape = x0.shape
        x_t = self.sqrt_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x0 + \
            self.sqrt_one_minus_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * noise

        return x_t
    
    def predict_v(self, x0, t, noise):
        B, *x_shape = x0.shape
        return self.sqrt_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * noise - \
            self.sqrt_one_minus_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x0
    
    def predict_x0_from_v(self, x_t, t, v):
        B, *x_shape = x_t.shape
        return self.sqrt_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.sqrt_one_minus_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * v
            
    def predict_x0_from_noise(self, x_t, t, noise):
        B, *x_shape = x_t.shape
        return self.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * noise
    
    def predict_noise_from_x0(self, x_t, t, x0):
        B, *x_shape = x_t.shape
        return (self.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x_t - x0) / \
                self.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape)))
                
    def forward(self, data: Dict) -> torch.Tensor:
        ret = dict()
        condtion = self.eps_model.condition(data)
        if self.loss_weight[5] > 0 and self.trunc_use and self.trunc_type != "anchor":
            cls_output = self.eps_model.classify(condtion)
            if self.class_loss_type == 'kl_div':
                trajlib_loss = data['primitive_loss'] # B x N; B batch size, N number of primitives
                T = 1.0 # temperature for 
                P = F.softmax(-trajlib_loss / T, dim=1)
                Q = F.softmax(cls_output, dim=1)
                kl_loss = F.kl_div(Q.log(), P, reduction='batchmean')
                ret['classify'] = kl_loss 
            else:
                ret['classify'] = self.criterion_class(cls_output, data['min_idx'].view(-1))
            ret['classify'] *= self.loss_weight[5]
        B = data['x'].shape[0]

        ## randomly sample timesteps
        if self.rand_t_type == 'all':
            ts = torch.randint(0, self.timesteps, (B, ), device=self.device).long()
        elif self.rand_t_type == 'half':
            ts = torch.randint(0, self.timesteps, ((B + 1) // 2, ), device=self.device)
            if B % 2 == 1:
                ts = torch.cat([ts, self.timesteps - ts[:-1] - 1], dim=0).long()
            else:
                ts = torch.cat([ts, self.timesteps - ts - 1], dim=0).long()
        else:
            raise Exception('Unsupported rand ts type.')
        
        if self.trunc_use:
            ts = torch.randint(0, self.trunc_steps, (B, ), device=self.device).long()
        
        ## generate Gaussian noise
        noise = torch.randn_like(data['x'], device=self.device)

        ## calculate x_t, forward diffusion process
        if self.trunc_use:
            if self.trunc_type == "gt":
                x_t = self.q_sample(x0=data['primitive'], t=ts, noise=noise)
            elif self.trunc_type == "trunc":
                prob = torch.softmax(cls_output, dim=1).transpose(0, 1)
                libnum = self.trajlib.shape[0]
                trajlib_all = self.trajlib.to(self.device).unsqueeze(1).repeat(1, B, 1, 1).reshape(B*libnum, noise.shape[1], noise.shape[2])
                trajlib_all = (trajlib_all - data['mu'].repeat(libnum, 1, 1)) / data['std'].repeat(libnum, 1, 1)
                ts_all = ts.repeat(libnum)
                noise_all = noise.repeat(libnum, 1, 1)
                x_t = self.q_sample(x0=trajlib_all, t=ts_all, noise=noise_all)
            elif self.trunc_type == "anchor":
                libnum = self.trajlib.shape[0]
                trajlib_all = self.trajlib.to(self.device).unsqueeze(1).repeat(1, B, 1, 1).reshape(B*libnum, noise.shape[1], noise.shape[2])
                trajlib_all = (trajlib_all - data['mu'].repeat(libnum, 1, 1)) / data['std'].repeat(libnum, 1, 1)
                ts_all = ts.repeat(libnum)
                noise_all = torch.randn_like(trajlib_all, device=self.device)
                x_t = self.q_sample(x0=trajlib_all, t=ts_all, noise=noise_all)
        else:
            x_t = self.q_sample(x0=data['x'], t=ts, noise=noise)
        
        x_t = self.apply_observation(x_t, data)
        
        if self.objective == "noise":
            target = noise
        elif self.objective == "x0":
            target = data['x']
        elif self.objective == "v":
            target = self.predict_v(x0=data['x'], t=ts, noise=noise)
        else:
            raise Exception('Unsupported objective type.')

        ## calculate loss
        if self.loss_weight[0] > 0:
            if self.trunc_use and self.trunc_type == "trunc":
                output, _ = self.eps_model(x_t, ts_all, condtion.repeat(libnum, 1, 1))
                rec_output = []
                for i in range(B):
                    rec_output.append(output.view(libnum, B, target.shape[1], target.shape[2])[data['min_idx'][i], i:i+1])
                rec_output = torch.cat(rec_output, dim=0)
                ret['rec'] = self.criterion(rec_output, target) * self.loss_weight[0]
                de_traj = self.transer.detrans((output, data["startgoal"][:, 7:10].detach().repeat(libnum, 1)))
                de_gt = self.transer.detrans((data['x'], data["startgoal"][:, 7:10])).detach().repeat(libnum, 1, 1)
                data["sdf"] = data["sdf"].repeat(libnum, 1, 1, 1)
                ret['sdf'] = (SafeCost(de_traj, data, redution="none").reshape(libnum, B) * prob).sum(dim=0).mean() * \
                    self.loss_weight[1] if self.loss_weight[1] > 0 else torch.as_tensor(0.0).to(de_traj.device)
                ret['vel'] = (VelCost(de_traj, data, 'none').reshape(libnum, B) * prob).sum(dim=0).mean() * \
                    self.loss_weight[2] if self.loss_weight[2] > 0 else torch.as_tensor(0.0).to(de_traj.device)
                ret['smooth'] = (NormedSmoothLoss(de_traj, de_gt, "none").reshape(libnum, B) * prob).sum(dim=0).mean() * \
                    self.loss_weight[3] if self.loss_weight[3] > 0 else torch.as_tensor(0.0).to(de_traj.device)
                ret['arc'] = (UniArcLoss(de_traj, de_gt, "none").reshape(libnum, B) * prob).sum(dim=0).mean() * \
                    self.loss_weight[4] if self.loss_weight[4] > 0 else torch.as_tensor(0.0).to(de_traj.device)
                ret['eearc']=(UniEEArcLoss(de_traj, de_gt, "none").reshape(libnum, B) * prob).sum(dim=0).mean() * \
                    self.loss_weight[6] if self.loss_weight[6] > 0 else torch.as_tensor(0.0).to(de_traj.device)
            if self.trunc_use and self.trunc_type == "anchor":
                output, anchor_prob = self.eps_model(x_t, ts_all, condtion.repeat(libnum, 1, 1))
                target_classes_onehot = torch.zeros([libnum, B],
                                                    dtype=anchor_prob.dtype,
                                                    layout=anchor_prob.layout,
                                                    device=anchor_prob.device)
                target_classes_onehot.scatter_(0, data['min_idx'].unsqueeze(0), 1)
                ret['classify'] = self.loss_weight[5] * self.criterion_class(anchor_prob, target_classes_onehot)

                pred_traj = output.view(libnum, B, target.shape[1], target.shape[2])
                anchor_idx = data['min_idx'].clone()
                anchor_idx = anchor_idx[None, ..., None, None].repeat(1, 1, target.shape[1], target.shape[2])
                best_reg = torch.gather(pred_traj, 0, anchor_idx).squeeze(0)

                ret['rec'] = self.criterion(best_reg, target) * self.loss_weight[0]
                de_traj = self.transer.detrans((best_reg, data["startgoal"][:, 7:10]))
                de_gt = self.transer.detrans((data['x'], data["startgoal"][:, 7:10])).detach()
                ret['sdf'] = SafeCost(de_traj, data) * self.loss_weight[1] if self.loss_weight[1] > 0 else torch.as_tensor(0.0).to(de_traj.device)
                ret['vel'] = VelCost(de_traj, data) * self.loss_weight[2] if self.loss_weight[2] > 0 else torch.as_tensor(0.0).to(de_traj.device)
                ret['smooth'] = NormedSmoothLoss(de_traj, de_gt) * self.loss_weight[3] if self.loss_weight[3] > 0 else torch.as_tensor(0.0).to(de_traj.device)
                ret['arc'] = UniArcLoss(de_traj, de_gt) * self.loss_weight[4] if self.loss_weight[4] > 0 else torch.as_tensor(0.0).to(de_traj.device)
                ret['eearc']=UniEEArcLoss(de_traj, de_gt) * self.loss_weight[6] if self.loss_weight[6] > 0 else torch.as_tensor(0.0).to(de_traj.device)
            else:
                output, _ = self.eps_model(x_t, ts, condtion)
                ret['rec'] = self.criterion(output, target) * self.loss_weight[0]
                de_traj = self.transer.detrans((output, data["startgoal"][:, 7:10]))
                de_gt = self.transer.detrans((data['x'], data["startgoal"][:, 7:10])).detach()
                ret['sdf'] = SafeCost(de_traj, data) * self.loss_weight[1] if self.loss_weight[1] > 0 else torch.as_tensor(0.0).to(de_traj.device)
                ret['vel'] = VelCost(de_traj, data) * self.loss_weight[2] if self.loss_weight[2] > 0 else torch.as_tensor(0.0).to(de_traj.device)
                ret['smooth'] = NormedSmoothLoss(de_traj, de_gt) * self.loss_weight[3] if self.loss_weight[3] > 0 else torch.as_tensor(0.0).to(de_traj.device)
                ret['arc'] = UniArcLoss(de_traj, de_gt) * self.loss_weight[4] if self.loss_weight[4] > 0 else torch.as_tensor(0.0).to(de_traj.device)
                ret['eearc']=UniEEArcLoss(de_traj, de_gt) * self.loss_weight[6] if self.loss_weight[6] > 0 else torch.as_tensor(0.0).to(de_traj.device)
            
        loss = 0.0
        for key in ret:
            loss += ret[key]
            
        ret['loss'] = loss
        # return ret, target, output
        return ret
    
    def model_predict(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> Tuple:
        """ Get and process model prediction

        $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        
        Return:
            The predict target `(pred_noise, pred_x0)`, currently we predict the noise, which is as same as DDPM
        """
        B, *x_shape = x_t.shape

        #! NOTE: check the usage. 
        clip_x_start = False
        rederive_pred_noise = False
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        model_output, anchor_prob = self.eps_model(x_t, t, cond)
        if self.objective == "noise":
            pred_noise = model_output
            pred_x0 = self.predict_x0_from_noise(x_t, t, pred_noise)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_x0(x_t, t, pred_x0) 
        elif self.objective == "x0":
            pred_x0 = model_output
            pred_x0 = maybe_clip(pred_x0)
            pred_noise = self.predict_noise_from_x0(x_t, t, pred_x0)
        elif self.objective == "v":
            pred_v = model_output
            pred_x0 = self.predict_x0_from_v(x_t, t, pred_v)
            pred_x0 = maybe_clip(pred_x0)
            pred_noise = self.predict_noise_from_x0(x_t, t, pred_x0)
            
        return pred_noise, pred_x0, anchor_prob
    
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> Tuple:
        """ Calculate the mean and variance, we adopt the following first equation.

        $\tilde{\mu} = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}x_0$
        $\tilde{\mu} = \frac{1}{\sqrt{\alpha}_t}(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        
        Return:
            (model_mean, posterior_variance, posterior_log_variance)
        """
        B, *x_shape = x_t.shape

        ## predict noise and x0 with model $p_\theta$
        # print(x_t.shape, cond.shape)
        pred_noise, pred_x0, _ = self.model_predict(x_t, t, cond)

        ## calculate mean and variance
        model_mean = self.posterior_mean_coef1[t].reshape(B, *((1, ) * len(x_shape))) * pred_x0 + \
            self.posterior_mean_coef2[t].reshape(B, *((1, ) * len(x_shape))) * x_t
        posterior_variance = self.posterior_variance[t].reshape(B, *((1, ) * len(x_shape)))
        posterior_log_variance = self.posterior_log_variance_clipped[t].reshape(B, *((1, ) * len(x_shape))) # clipped variance

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int, data: Dict) -> torch.Tensor:
        """ One step of reverse diffusion process

        $x_{t-1} = \tilde{\mu} + \sqrt{\tilde{\beta}} * z$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            data: data dict that provides original data and computed conditional feature

        Return:
            Predict data in the previous step, i.e., $x_{t-1}$
        """
        B, *_ = x_t.shape
        batch_timestep = torch.full((B, ), t, device=self.device, dtype=torch.long)

        if 'cond' in data:
            ## use precomputed conditional feature
            cond = data['cond']
        else:
            ## recompute conditional feature every sampling step
            cond = self.eps_model.condition(data)
        model_mean, model_variance, model_log_variance = self.p_mean_variance(x_t, batch_timestep, cond)
        
        if self.add_guidance and t < self.guidin_time and t > 0:
            print("begin add guidance")
            for i in range(self.guidin_num_per_time):
                gradient = self.guidance.gradients(model_mean, data)
                model_mean = model_mean - gradient
                # model_mean = model_mean - model_variance * gradient

        noise = torch.randn_like(x_t) if t > 0 else 0. # no noise if t == 0
        
        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred_x
    
    @torch.no_grad()
    def p_sample_batch(self, x_t: torch.Tensor, ts: torch.Tensor, data: Dict) -> torch.Tensor:
        """ One step of reverse diffusion process

        $x_{t-1} = \tilde{\mu} + \sqrt{\tilde{\beta}} * z$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            data: data dict that provides original data and computed conditional feature

        Return:
            Predict data in the previous step, i.e., $x_{t-1}$
        """
        B, *_ = x_t.shape
        batch_timestep = ts.long()

        if 'cond' in data:
            ## use precomputed conditional feature
            cond = data['cond']
        else:
            ## recompute conditional feature every sampling step
            cond = self.eps_model.condition(data)
        model_mean, model_variance, model_log_variance = self.p_mean_variance(x_t, batch_timestep, cond)
        
        # noise = torch.randn_like(x_t) if t > 0 else 0. # no noise if t == 0
        noise = torch.randn_like(x_t)
        
        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred_x
    
    @torch.no_grad()
    def p_sample_ddim(self, data: Dict, trajlib: torch.Tensor) -> torch.Tensor:
        """ Reverse diffusion process loop, iteratively sampling

        Args:
            data: test data, data['x'] gives the target data shape
        
        Return:
            Sampled data, <B, T, ...>
        """

        if self.trunc_use:
            times = torch.linspace(-1, self.trunc_steps-1, steps=self.ddimsteps + 1)
        else:
            # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            times = torch.linspace(-1, self.timesteps - 1, steps=self.ddimsteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        _b = data['x'].shape[0]
        ## precompute conditional feature, which will be used in every sampling step
        condition = self.eps_model.condition(data)
        prob = self.eps_model.classify(condition)
        prob = torch.softmax(prob, dim=1)
        prbs, pred_idxs = torch.topk(prob[0], _b)

        if _b > condition.shape[0]:
            condition = condition.expand((_b, -1, -1))
            # prob = prob.expand((_b, -1))
            # prbs = prbs.expand((_b))
            # pred_idxs = pred_idxs.expand(_b)

        predict_prims = trajlib[pred_idxs.long().cpu()].to(condition.device)
        predict_prims = (predict_prims - data['mu']) / data['std']
        
        data['cond'] = condition

        x_t = torch.randn_like(data['x'], device=self.device)
        # self.NOISE = x_t
        if self.trunc_use:
            ts = torch.ones((data['x'].shape[0], ), device=self.device).long() * times[0]
            if self.trunc_type == "gt":
                x_t = self.q_sample(x0=data['primitive'], t=ts, noise=x_t)
            elif self.trunc_type == "trunc":
                x_t = self.q_sample(x0=predict_prims, t=ts, noise=x_t)
                # self.PRIMS = predict_prims
            elif self.trunc_type == "anchor":
                Bs = x_t.shape[0]
                libnum = self.trajlib.shape[0]
                trajlib_all = self.trajlib.to(self.device).unsqueeze(1).repeat(1, Bs, 1, 1).reshape(Bs*libnum, x_t.shape[1], x_t.shape[2])
                trajlib_all = (trajlib_all - data['mu'].repeat(libnum, 1, 1)) / data['std'].repeat(libnum, 1, 1)
                self.ANCHOR_PRIMS = trajlib_all
                x_t = torch.randn_like(trajlib_all, device=self.device)
                self.ANCHOR_NOISE = x_t.clone()
                data['cond'] = data['cond'].repeat(libnum, 1, 1)
                ts_all = ts.repeat(libnum)
                trajlib_all, x_t = trajlib_all.half(), x_t.half()
                x_t = self.q_sample(x0=trajlib_all, t=ts_all, noise=x_t)
            
        x_start = None
        all_x_t = [x_t]
        # for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        for time, time_next in time_pairs:

            B, *_ = x_t.shape
            batch_timestep = torch.full((B, ), time, device=self.device, dtype=torch.long)

            if 'cond' in data:
                ## use precomputed conditional feature
                cond = data['cond']
            else:
                ## recompute conditional feature every sampling step
                cond = self.eps_model.condition(data)
            
            # x_t = self.apply_observation(x_t, data)
            pred_noise, x_start, anchor_prob = self.model_predict(x_t, batch_timestep, cond)
            # x_start = self.apply_observation(x_start, data)

            if time_next < 0:
                x_t = x_start
                # x_t = self.apply_observation(x_t, data)
                all_x_t.append(x_t.clone())
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.ddim_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_t)

            x_t = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
            # x_t = self.apply_observation(x_t, data)
            all_x_t.append(x_t.clone())

        # self.ANCHOR_FULL = [_x.clone() for _x in all_x_t]

        if self.trunc_use and self.trunc_type == "anchor":
            prbs, pred_idx = torch.topk(anchor_prob[:, 0], Bs)
            self.PRED_IDX = pred_idx.clone()
            for i in range(len(all_x_t)):
                anchor_idx = pred_idx.clone()
                anchor_idx = anchor_idx[None, ..., None, None].repeat(1, 1, x_t.shape[1], x_t.shape[2])
                best_reg = torch.gather(all_x_t[i].reshape(libnum, Bs, x_t.shape[1], x_t.shape[2]), 0, anchor_idx).squeeze(0)
                all_x_t[i] = best_reg

        all_x_t[-1] = self.apply_observation(all_x_t[-1], data)

        return torch.stack(all_x_t, dim=1), predict_prims, prbs
    
    @torch.no_grad()
    def p_sample_loop(self, data: Dict) -> torch.Tensor:
        """ Reverse diffusion process loop, iteratively sampling

        Args:
            data: test data, data['x'] gives the target data shape
        
        Return:
            Sampled data, <B, T, ...>
        """
        x_t = torch.randn_like(data['x'], device=data['x'].device)
        ## apply observation to x_t
        # x_t = self.apply_observation(x_t, data)
        
        ## precompute conditional feature, which will be used in every sampling step
        condition = self.eps_model.condition(data)
        _b = data['x'].shape[0]
        if _b > condition.shape[0]:
            condition = condition.expand((_b, -1, -1))
        data['cond'] = condition

        ## iteratively sampling
        all_x_t = [x_t]
        for t in tqdm(reversed(range(0, self.timesteps)), desc = 'sampling loop time step', total = self.timesteps):
            x_t = self.p_sample(x_t, t, data)
            ## apply observation to x_t
            # x_t = self.apply_observation(x_t, data)
            
            all_x_t.append(x_t)
        all_x_t[-1] = self.apply_observation(all_x_t[-1], data)
        return torch.stack(all_x_t, dim=1), None, None
    
    @torch.no_grad()
    def sample(self, trajlib: torch.Tensor, data: Dict) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition
        In this method, the sampled results are unnormalized and converted to absolute representation.

        Args:
            data: test data, data['x'] gives the target data shape
            k: the number of sampled data
        
        Return:
            Sampled results, the shape is <B, k, T, ...>
        """
        ksamples = []
        if self.ddim_use or self.trunc_use:
            sampled, predict_prims, prbs = self.p_sample_ddim(data, trajlib)
        else:
            sampled, predict_prims, prbs = self.p_sample_loop(data)
        ksamples.append(sampled)
        
        ksamples = torch.stack(ksamples, dim=1)
        
        ## for sequence, normalize and convert repr
        # TODO: use self.normer
        if 'normalizer' in data and data['normalizer'] is not None:
            O = 0
            if self.has_observation and 'start' in data:
                ## the start observation frames are replace during sampling
                _, O, _ = data['start'].shape
            ksamples[..., O:, :] = data['normalizer'].unnormalize(ksamples[..., O:, :])
        
        return ksamples, predict_prims, prbs
    
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

    @torch.no_grad()
    def p_sample_ddim_trace(self, data: Dict, trajlib: torch.Tensor) -> torch.Tensor:
        """ Reverse diffusion process loop, iteratively sampling

        Args:
            data: test data, data['x'] gives the target data shape
        
        Return:
            Sampled data, <B, T, ...>
        """

        if self.trunc_use:
            times = torch.linspace(-1, self.trunc_steps-1, steps=self.ddimsteps + 1)
        else:
            # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            times = torch.linspace(-1, self.timesteps - 1, steps=self.ddimsteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        ## precompute conditional feature, which will be used in every sampling step
        condition = self.eps_model.condition(data)
        prob = self.eps_model.classify(condition)
        prob = torch.softmax(prob, dim=1)
        prbs, pred_idxs = torch.topk(prob[0], prob.shape[0])

        _b = data['x'].shape[0]
        if _b > condition.shape[0]:
            condition = condition.expand((_b, -1, -1))
            prob = prob.expand((_b, -1))
            prbs = prbs.expand((_b))
            pred_idxs = pred_idxs.expand(_b)

        predict_prims = trajlib[pred_idxs.long().cpu()].to(condition.device)
        predict_prims = (predict_prims - data['mu']) / data['std']

        data['cond'] = condition

        x_t = torch.randn_like(data['x'], device=self.device)
        if self.trunc_use:
            ts = torch.ones((data['x'].shape[0], ), device=self.device).long() * times[0]
            if self.trunc_type == "gt":
                x_t = self.q_sample(x0=data['primitive'], t=ts, noise=x_t)
            elif self.trunc_type == "trunc":
                x_t = self.q_sample(x0=predict_prims, t=ts, noise=x_t)
            elif self.trunc_type == "anchor":
                Bs = x_t.shape[0]
                libnum = self.trajlib.shape[0]
                trajlib_all = self.trajlib.to(self.device).unsqueeze(1).repeat(1, Bs, 1, 1).reshape(Bs*libnum, x_t.shape[1], x_t.shape[2])
                trajlib_all = (trajlib_all - data['mu'].repeat(libnum, 1, 1)) / data['std'].repeat(libnum, 1, 1)
                x_t = torch.randn_like(trajlib_all, device=self.device)
                data['cond'] = data['cond'].repeat(libnum, 1, 1)
                ts_all = ts.repeat(libnum)
                x_t = self.q_sample(x0=trajlib_all, t=ts_all, noise=x_t)
            
        x_start = None
        all_x_t = [x_t]
        # for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        for time, time_next in time_pairs:

            B, *_ = x_t.shape
            batch_timestep = torch.full((B, ), time, device=self.device, dtype=torch.long)

            if 'cond' in data:
                ## use precomputed conditional feature
                cond = data['cond']
            else:
                ## recompute conditional feature every sampling step
                cond = self.eps_model.condition(data)

            # x_t = self.apply_observation(x_t, data)
            pred_noise, x_start, anchor_prob = self.model_predict(x_t, batch_timestep, cond)
            # x_start = self.apply_observation(x_start, data)

            if time_next < 0:
                x_t = x_start
                # x_t = self.apply_observation(x_t, data)
                all_x_t.append(x_t.clone())
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.ddim_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_t)

            x_t = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
            # x_t = self.apply_observation(x_t, data)
            all_x_t.append(x_t.clone())

        # self.ANCHOR_FULL = all_x_t

        if self.trunc_use and self.trunc_type == "anchor":
            prbs, pred_idx = torch.topk(anchor_prob[:, 0], Bs)
            for i in range(len(all_x_t)):
                anchor_idx = pred_idx.clone()
                anchor_idx = anchor_idx[None, ..., None, None].repeat(1, 1, x_t.shape[1], x_t.shape[2])
                best_reg = torch.gather(all_x_t[i].reshape(libnum, Bs, x_t.shape[1], x_t.shape[2]), 0, anchor_idx).squeeze(0)
                all_x_t[i] = best_reg

        all_x_t[-1] = self.apply_observation(all_x_t[-1], data)

        return torch.stack(all_x_t, dim=1), predict_prims, prbs


    @torch.no_grad()
    def sample_trace(self, trajlib: torch.Tensor, data: Dict) -> torch.Tensor:
        """sample without if...else..., to be traced

        Args:
            trajlib (torch.Tensor): _description_
            data (Dict): _description_

        Returns:
            torch.Tensor: _description_
        """
        ksamples = []
        sampled, predict_prims, prbs = self.p_sample_ddim(data, trajlib)

        ksamples.append(sampled)
        ksamples = torch.stack(ksamples, dim=1)

        return ksamples, predict_prims, prbs
