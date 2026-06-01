import torch
import numpy as np
from torch import vmap
from tqdm import tqdm
from moma_param import MomaParam
moma_utils = MomaParam()

def update_grid_map(grid_map, ball):
    voxel_size = 0.1
    x, y, z, r = ball // voxel_size
    x += 50
    y += 50

    x_min, x_max = x - r, x + r
    y_min, y_max = y - r, y + r
    z_min, z_max = z - r, z + r

    x_min = torch.clamp(x_min, 0, 100 - 1).long()
    x_max = torch.clamp(x_max, 0, 100 - 1).long()
    y_min = torch.clamp(y_min, 0, 100 - 1).long()
    y_max = torch.clamp(y_max, 0, 100 - 1).long()
    z_min = torch.clamp(z_min, 0, 16 - 1).long()
    z_max = torch.clamp(z_max, 0, 16 - 1).long()

    for xi in range(x_min, x_max + 1):
        for yi in range(y_min, y_max + 1):
            for zi in range(z_min, z_max + 1):
                dx = xi - x
                dy = yi - y
                dz = zi - z
                distance = dx ** 2 + dy ** 2 + dz ** 2
                if distance <= r ** 2:
                    grid_map[xi, yi, zi] = True

    return

# trajs: N x wps_num x 13 x 4
def getDscore(trajs):
    d = 0.0
    params = {}
    params["map_size_x"] = 10.0
    params["map_size_y"] = 10.0
    params["map_size_z"] = 1.6
    params["resolution"] = 0.1
    params["origin_x"] = -5.0
    params["origin_y"] = -5.0
    params["origin_z"] = 0.0
    
    grid_maps = torch.zeros((trajs.shape[0], 100, 100, 16), dtype=torch.bool).cuda()
    for i in range(trajs.shape[0]):
        obs = trajs[i].flatten().cpu().numpy()
        grid_map = np.zeros(100*100*16, dtype=np.int32)
        moma_utils.getBallsGrids(params, obs, grid_map)
        grid_maps[i] = torch.from_numpy(grid_map).reshape(100, 100, 16).bool().cuda()
        # for j in range(trajs.shape[1]):
        #     for k in range(trajs.shape[2]):
        #         update_grid_map(grid_maps[i], trajs[i,j,k])
    
    for i in range(trajs.shape[0]):
        grid_a = grid_maps[i]
        grid_b = torch.cat([grid_maps[:i], grid_maps[i+1:]], dim=0).any(dim=0)
        
        intersection = (grid_a & grid_b).sum().item()
        union = grid_a.sum().item() + grid_b.sum().item() - intersection
        iou = intersection / union
        d += iou
        
    d /= trajs.shape[0]
    return 1.0 - d

