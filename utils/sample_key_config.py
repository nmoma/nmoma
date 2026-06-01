"""
This file implements key configuration sampling method from PRESTO (https://arxiv.org/pdf/2409.16012)
It might be slow for large datasets, since it detects collision with sdf of all datapoints.

TODO: Nice to have: detect collision with subset only to speed up sampling.
"""
import torch
import random
import numpy as np
from torch.utils.data import DataLoader


from utils.data_loader import MomaDataset, collate_fn_4ptrans
from loss import isStateFree
from utils.utils import getColliPts

if not torch.cuda.is_available():
    raise Exception("CUDA is not available. Please check your setup.")

def sample_key_config(
        filepath : str,
        num_config : int,
        min_cspace_dist=0.5,
        min_wspace_dist=0.2,
        collision_prop_bound=0.3,
        verbose=False):
    """sample key configurations from dataset

    Args:
        filepath (str): path to dataset file.
        num_config (int): number of key configurations to sample.
        min_cspace_dist (float, optional): minimum cspace distance between configs. Defaults to 0.0.
        min_wspace_dist (float, optional): minimum wspace distance between configs. Defaults to 0.0.
        collision_prop_bound (float, optional): _description_. Defaults to 0.3.
        verbose (bool, optional): Whether to print progress. Defaults to False.

    Returns:
        key_config (np.ndarray): Numpy array of key configurations of shape (N, 11).
    """
    dataset = MomaDataset(filepath=filepath, norm_method="shoot")
    colli_dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        pin_memory=True,
        shuffle=True,
        num_workers=4
    )

    iteration_cnt = 0
    key_config_list = [] # N x 11
    colli_pts_list = []  # N x 13 x 3
    while len(key_config_list) < num_config:
        # sample config from dataset

        data = random.choice(dataset)
        data['x'].unsqueeze_(0)
        data['startgoal'].unsqueeze_(0)

        data['x'] = dataset.transer.detrans((data['x'], data['startgoal'][:, 7:10]))
        wp_num = data['x'].shape[1]
        config = data['x'][0][random.randint(0, wp_num-1)].cuda()


        # # compute wspace and cspace distance
        wspace_dist = torch.inf
        cspace_dist = torch.inf

        colli_pts = getColliPts(config.unsqueeze(0)).cuda() # 1 x 13 x 3
        if len(key_config_list) > 0:
            prev_config = torch.stack(key_config_list).cuda() # N x 11
            prev_colli_pts = torch.stack(colli_pts_list, dim=0).squeeze(1).cuda() # N x 13 x 3

            # min across all previous configs, max across all collision points
            # ? seems to make sense
            tmp = torch.max(
                torch.norm(colli_pts.expand(len(key_config_list), -1, -1) - prev_colli_pts, dim=2), dim=1
            )

            wspace_dist = torch.min(tmp.values).detach()

            cspace_dist = torch.min(
                torch.norm(config.expand(len(key_config_list), -1) - prev_config, dim=1)
            ).detach()


        wspace_dist_flag = wspace_dist > min_wspace_dist
        cspace_dist_flag = cspace_dist > min_cspace_dist
        colli_prop_flag = False

        # check collision proportion, across all environments
        if wspace_dist_flag and cspace_dist_flag:
            total_num = 0
            free_num = 0
            for _d in colli_dataloader:
                N = _d['x'].shape[0]
                free_ = isStateFree(
                    moma_state=config.expand(N, -1).cuda(), \
                    sdf=_d['sdf'].flatten(start_dim=1).cuda(), \
                    safe_margin=1e-3).cpu()
                total_num += torch.numel(free_)
                free_num  += torch.count_nonzero(free_)
            colli_proportion = (total_num - free_num) / total_num
            colli_prop_flag = colli_proportion > collision_prop_bound and colli_proportion < (1 - collision_prop_bound)
        wspace_dist_flag = True
        cspace_dist_flag = True
        colli_prop_flag = True
        if wspace_dist_flag and cspace_dist_flag and colli_prop_flag:
            key_config_list.append(config.cpu())
            colli_pts_list.append(colli_pts)

        iteration_cnt += 1
        if verbose: print(f"Iteration {iteration_cnt}: {len(key_config_list)} sampled.")
    return np.stack([cfg.numpy() for cfg in key_config_list])

if __name__ == '__main__':
    num_config = 4096
    key_config = sample_key_config(filepath="../simple_10k.h5", num_config=num_config,
                                   collision_prop_bound=0.15,
                                   min_cspace_dist=0.1,
                                   min_wspace_dist=0.1, verbose=True)
    np.save(f"../kc_{num_config}.npy", key_config)
