import os
import torch
import random
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from modules.unet import UNetModel
from plot import Ploter
from modules.ddpm import DDPM
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from modules.guidance import SafeCost
import time

from einops import rearrange

from utils.mIoU import getDscore
from utils.utils import *
from utils.data_loader import MomaDataset, collate_fn_4ptrans


colli_radius = torch.tensor(getColliPtsRadius()).cuda()
def vis_traj(ax, pts, intvl=4, color='blue'):
    for i in range(0, pts.shape[0], intvl):
        joint = 0
        # Create the data points for the cylinder
        num_points = 5
        theta = np.linspace(0, 2 * np.pi, 8)
        z = np.linspace(0, MOMA_PARAM.chassis_height, num_points)
        theta_grid, z_grid = np.meshgrid(theta, z)

        # Convert theta and z to x and y, using the equation of a circle
        x_grid = MOMA_PARAM.chassis_colli_radius * np.cos(theta_grid) + np.array(pts[i, joint, 0])
        y_grid = MOMA_PARAM.chassis_colli_radius * np.sin(theta_grid) + np.array(pts[i, joint, 1])
        ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.2)

        for joint in range(1, 13):
            theta = np.linspace(0, 2 * np.pi, 4)
            phi = np.linspace(0, np.pi, 16)
            theta, phi = np.meshgrid(theta, phi)
            r = colli_radius[joint].cpu().numpy()
            x = r * np.sin(phi) * np.cos(theta) + np.array(pts[i, joint, 0])
            y = r * np.sin(phi) * np.sin(theta) + np.array(pts[i, joint, 1])
            z = r * np.cos(phi) + np.array(pts[i, joint, 2])
            ax.plot_surface(x, y, z, color=color, alpha=0.2)

def _set_submodule(root: torch.nn.Module, name: str, module: torch.nn.Module) -> None:
    parent_name, child_name = name.rsplit('.', 1)
    parent = root.get_submodule(parent_name)
    if isinstance(parent, torch.nn.Sequential) and child_name.isdigit():
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)

def _adapt_layernorms_to_ckpt(model: torch.nn.Module, saved_state_dict: dict) -> None:
    model_state_dict = model.state_dict()
    adapted = []
    adapted_modules = set()

    for saved_key, saved_value in saved_state_dict.items():
        key = saved_key[7:] if saved_key.startswith('module.') else saved_key
        if not key.endswith(('.weight', '.bias')):
            continue
        if key not in model_state_dict:
            continue
        if saved_value.shape == model_state_dict[key].shape:
            continue

        module_name, param_name = key.rsplit('.', 1)
        if module_name in adapted_modules:
            continue

        submodule = model.get_submodule(module_name)
        if not isinstance(submodule, torch.nn.LayerNorm):
            continue

        normalized_shape = tuple(saved_value.shape)
        device = submodule.weight.device if submodule.elementwise_affine else saved_value.device
        dtype = submodule.weight.dtype if submodule.elementwise_affine else saved_value.dtype
        new_layer = torch.nn.LayerNorm(
            normalized_shape,
            eps=submodule.eps,
            elementwise_affine=submodule.elementwise_affine,
        ).to(device=device, dtype=dtype)

        _set_submodule(model, module_name, new_layer)
        adapted.append((module_name, param_name, tuple(model_state_dict[key].shape), normalized_shape))
        adapted_modules.add(module_name)

def load_ckpt(model: torch.nn.Module, path: str, map_location="cuda:0") -> None:
    """ load ckpt for current model

    Args:
        model: current model
        path: save path
    """
    assert os.path.exists(path), 'Can\'t find provided ckpt.'

    saved_ckp = torch.load(path, map_location=map_location)
    print(f"Loaded ckpt epoch: {saved_ckp['epoch']}, step: {saved_ckp['step']}")
    saved_state_dict = saved_ckp['model']
    _adapt_layernorms_to_ckpt(model, saved_state_dict)
    model_state_dict = model.state_dict()

    loaded_keys = []
    skipped_shape_keys = []

    for key, current_value in model_state_dict.items():
        saved_value = None
        if key in saved_state_dict:
            saved_value = saved_state_dict[key]
        ## model is trained with ddm
        elif 'module.' + key in saved_state_dict:
            saved_value = saved_state_dict['module.' + key]

        if saved_value is None:
            continue

        if saved_value.shape != current_value.shape:
            skipped_shape_keys.append((key, tuple(saved_value.shape), tuple(current_value.shape)))
            continue

        model_state_dict[key] = saved_value
        loaded_keys.append(key)
        # logger.info(f'Load parameter {key} for current model.')

    model.load_state_dict(model_state_dict)
    print(f'Loaded {len(loaded_keys)}/{len(model_state_dict)} tensors from checkpoint.')
    if skipped_shape_keys:
        for key, ckpt_shape, model_shape in skipped_shape_keys:
            print(f'  {key}: ckpt {ckpt_shape} != model {model_shape}')

    params = []
    nparams = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
            nparams.append(p.nelement())
    print(f'{len(params)} parameters for optimization.')
    print(f'total model size is {sum(nparams)/1024/1024} mb.')

def main() -> None:
    """ training portal, train with multi gpus

    Args:
        cfg: configuration dict
    """

    cfg = OmegaConf.load("config/ddpm.yaml")

    logger.add(cfg.exp_dir + '/runtime.log')

    test_set = MomaDataset("replica1k.h5", cfg.data_normer, mustd_path="logs/mustd.npy")

    logger.info(f'Load dataset size: {len(test_set)}')

    data_loader = DataLoader(
        test_set,
        batch_size=1,
        collate_fn=collate_fn_4ptrans,
        pin_memory=True,
        drop_last = True,
        shuffle=True
    )

    ## create model and optimizer
    unet = UNetModel(cfg, slurm=False).cuda()
    model = DDPM(test_set.transer, unet, cfg.diffuser).cuda()
    print(cfg.ckpt_dir)

    load_ckpt(model, path=cfg.ckpt_dir + '/model.pth')
    # params
    nsample = 4
    plt_prim = False
    plt_sample = True

    colors = [(1, 0, 1), (0, 1, 1)]
    for i in range(nsample*2):
        colors.append((random.random(), random.random(), random.random()))
    model.eval()
    safe_costs = [0 for i in range(nsample*2+2)]

    for it, data in enumerate(data_loader):
        if it < 21:
            continue
        print("-------------------, it:", it)
        data_in = {}
        for key in data:
            if torch.is_tensor(data[key]):
                sample_shape = (nsample, )
                len_sh = len(data[key].shape)-1
                for i in range(len_sh):
                    sample_shape += (1, )
                data_in[key] = data[key].cuda().repeat(sample_shape)
        offset, count = [], 0
        for item in data_in['pos']:
            count += item.shape[0]
            offset.append(count)
        offset = torch.IntTensor(offset).to(data_in['pos'].device)
        data_in['offset'] = offset

        t0 = time.time()
        outputs, predict_prims, prbs = model.sample(test_set.trajlib, data_in)
        outputs = outputs[:, 0, -1]
        print("time consuming:", time.time()-t0, "s")

        sdf_points_world = data["pos"]
        wps = test_set.transer.detrans((data["x"], data["startgoal"][:, 7:10]))
        # prim = test_set.transer.detrans((data["primitive"], data["startgoal"][:, 7:10]))
        # predict_prims = test_set.transer.detrans((predict_prims, data_in["startgoal"][:, 7:10]))
        outputs = test_set.transer.detrans((outputs, data_in["startgoal"][:, 7:10]))

        # wps = torch.cat([wps.cuda(), outputs], dim=0)

        # wps_num =wps.shape[1]
        # colli_pts = torch.zeros((nsample, wps_num, 13, 4)).cuda()
        # colli_radius = torch.tensor(getColliPtsRadius()).cuda()
        # colli_pts[..., 3] = colli_radius.unsqueeze(0).unsqueeze(0).repeat(nsample, wps_num, 1)

        # mious = [0, 0]
        # # colli_pts[..., :3] = getColliPts(predict_prims.view(-1, 11)).view(nsample, wps_num, 13, 3)
        # # mious[0] += getDscore(colli_pts)
        # colli_pts[..., :3] = getColliPts(outputs.view(-1, 11)).view(nsample, wps_num, 13, 3)
        # mious[1] += getDscore(colli_pts)
        # print("dscore:", mious)

        # for i in range(nsample*2+2):
        #     safe_cost = SafeCost(wps[i].unsqueeze(0), data)
        #     safe_costs[i] += safe_cost.item()
        # continue

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        _pts = getColliPts(wps[0])
        _pts = _pts.cpu().numpy()
        _pts = rearrange(_pts, 'A B C -> (A B) C')
        print('_pts.shape', _pts.shape)
        # vis_traj(ax, _pts, intvl=4, color='red')
        ax.scatter(_pts[:, 0], _pts[:, 1], _pts[:, 2], color='black')

        _pts = getColliPts(outputs[0])
        _pts = _pts[::2, :].cpu().numpy()
        _pts = rearrange(_pts, 'A B C -> (A B) C')
        # vis_traj(ax, _pts, intvl=4, color='grey')
        ax.scatter(_pts[:, 0], _pts[:, 1], _pts[:, 2])

        _pts = getColliPts(outputs[1])
        _pts = _pts[::2, :].cpu().numpy()
        _pts = rearrange(_pts, 'A B C -> (A B) C')
        ax.scatter(_pts[:, 0], _pts[:, 1], _pts[:, 2])

        _pts = getColliPts(outputs[2])
        _pts = _pts[::2, :].cpu().numpy()
        _pts = rearrange(_pts, 'A B C -> (A B) C')
        ax.scatter(_pts[:, 0], _pts[:, 1], _pts[:, 2])

        _pts = getColliPts(outputs[3])
        _pts = _pts[::2, :].cpu().numpy()
        _pts = rearrange(_pts, 'A B C -> (A B) C')
        ax.scatter(_pts[:, 0], _pts[:, 1], _pts[:, 2])
        # vis_traj(ax, _pts, intvl=4, color='grey')
        # for i in range(0, _pts.shape[0], 8):
        #     joint = 0
        #     # Create the data points for the cylinder
        #     num_points = 5
        #     theta = np.linspace(0, 2 * np.pi, 16)
        #     z = np.linspace(0, MOMA_PARAM.chassis_height, num_points)
        #     theta_grid, z_grid = np.meshgrid(theta, z)

        #     # Convert theta and z to x and y, using the equation of a circle
        #     x_grid = MOMA_PARAM.chassis_colli_radius * np.cos(theta_grid) + np.array(_pts[i, joint, 0])
        #     y_grid = MOMA_PARAM.chassis_colli_radius * np.sin(theta_grid) + np.array(_pts[i, joint, 1])
        #     ax.plot_surface(x_grid, y_grid, z_grid, color='blue', alpha=0.2)

        #     for joint in range(1, 13):
        #         theta = np.linspace(0, 2 * np.pi, 4)
        #         phi = np.linspace(0, np.pi, 16)
        #         theta, phi = np.meshgrid(theta, phi)
        #         r = colli_radius[joint].cpu().numpy()
        #         x = r * np.sin(phi) * np.cos(theta) + np.array(_pts[i, joint, 0])
        #         y = r * np.sin(phi) * np.sin(theta) + np.array(_pts[i, joint, 1])
        #         z = r * np.cos(phi) + np.array(_pts[i, joint, 2])
        #         ax.plot_surface(x, y, z, color='blue', alpha=0.2)

        #! xulong: points obstacles
        print(data['startgoal'].shape)
        sdf_points_worldn = sdf_points_world[0]
        print(sdf_points_worldn.shape)
        sdf_points_worldn = unrotate_points(sdf_points_worldn, data['startgoal'][0])
        sdf_points_worldn = sdf_points_worldn.cpu().numpy()

        # if len(sdf_points_worldn) > 2000:
        #     indices = np.random.choice(len(sdf_points_worldn), 2000, replace=False)
        #     subsampled_points = sdf_points_worldn[indices]
        # else:
        #     subsampled_points = sdf_points_worldn
        subsampled_points = sdf_points_worldn
        ax.scatter(subsampled_points[:, 0], subsampled_points[:, 1], subsampled_points[:, 2], c='red', marker='o')

        # points_wps_cuda, _ = get_robo_tensor(wps.cuda())
        # print("-------------------")
        # print("primitive prbs: ", prbs)

        # for i in range(nsample*2+2):
        #     # 绘制 wps
        #     points_wps = points_wps_cuda.cpu().numpy()[i]
        #     print(f"the {i}-th sample safe_cost: {safe_costs[i]}")
        #     if i == 0:
        #         ax.scatter([0], [0], [0], color=colors[i], label=f'Ground Truth')
        #     elif i == 1 and plt_prim:
        #         ax.scatter([0], [0], [0], color=colors[i], label=f'Primitive')
        #     elif i < nsample + 2 and plt_prim:
        #         ax.scatter([0], [0], [0], color=colors[i], label=f'Predicted Prim {i-1}')
        #     elif i > nsample + 1 and plt_sample:
        #         ax.scatter([0], [0], [0], color=colors[i], label=f'Sample {i-nsample-1}')

        #     if i > 0 and i < nsample + 2 and not plt_prim:
        #         continue
        #     if i >= nsample + 2 and not plt_sample:
        #         continue

        #     for j in range(points_wps.shape[0]):
        #         ax.plot(points_wps[j, :, 0], points_wps[j, :, 1], points_wps[j, :, 2], c=colors[i], marker='o')
        #         if i == nsample + 2:
        #             ax.text(points_wps[j, -1, 0], points_wps[j, -1, 1], points_wps[j, -1, 2] + 1, str(j+1), color='red')

        #! wong: raw box
        # for box in data['raw_boxes'][0]:
        #     x,y,z, w,l,h, cos_theta, sin_theta = box

        #     vertices = np.array([
        #         [0, 0, 0],  # Vertex 0
        #         [1, 0, 0],  # Vertex 1
        #         [1, 1, 0],  # Vertex 2
        #         [0, 1, 0],  # Vertex 3
        #         [0, 0, 1],  # Vertex 4
        #         [1, 0, 1],  # Vertex 5
        #         [1, 1, 1],  # Vertex 6
        #         [0, 1, 1]   # Vertex 7
        #     ])
        #     vertices = vertices * np.array([[w, l, h]])
        #     rotation_matrix = np.array([
        #         [cos_theta, -sin_theta, 0],
        #         [sin_theta, cos_theta, 0],
        #         [0, 0, 1]
        #     ])
        #     vertices = vertices @ rotation_matrix.T  # Rotate the vertices

        #     vertices += np.array([[x, y, z]])


        #     # Define the 12 edges (pairs of vertex indices)
        #     edges = [
        #         [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        #         [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        #         [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        #     ]

        #     # Plot the edges
        #     for edge in edges:
        #         ax.plot3D(*vertices[edge].T, color='blue', linewidth=1)

        ax.set_aspect('equal', 'box')
        ax.view_init(elev=-90, azim=0)
        ax.legend()
        manager = plt.get_current_fig_manager()
        manager.resize(2048, 1660)

        def on_key_event(event):
            if event.key == 'q':
                plt.close()
                sys.exit()
        plt.gcf().canvas.mpl_connect('key_press_event', on_key_event)
        plt.show()

    print("-")
    for i in range(nsample*2+2):
        print(f"the {i}-th sample safe_cost: {safe_costs[i]}")

if __name__ == '__main__':
    ## set random seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()
