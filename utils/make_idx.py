from utils.data_loader import MomaDataset
import numpy as np
from tqdm import tqdm
from utils.utils import *
from loss import *
from einops import rearrange

import math
from mpi4py import MPI
import h5py

def find_min_idx_se2(data, trajlib, transer):
    assert trajlib.shape[-1] == 4, "trajlib should be in SE(2) format"
    wps = data["x"] * transer.std + transer.mu
    wps = rearrange(wps, 'w x -> 1 w x')
    wps = wps.to(trajlib.device)

    wps_se2 = wps[..., :4] # (1x64x4)
    distances = (wps_se2 - trajlib[..., :4]) # (32 x 64 x 4)

    distances = distances.norm(dim=-1)
    distances = distances.mean(dim=-1)
    return torch.argmin(distances)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str)
    parser.add_argument('-i', '--idx_file', type=str)
    parser.add_argument('--norm_method', type=str, default="shoot")
    parser.add_argument('--lib_path', type=str, default="logs/trajlib.npy")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    ds = MomaDataset(filepath=args.filepath,
                    norm_method=args.norm_method,
                    lib_path=args.lib_path,
                    mustd_path='logs/mustd.npy')
    # print(ds[0]['x'].shape)
    trajlib = np.load(args.lib_path, allow_pickle=True)
    trajlib = torch.as_tensor(trajlib).float()
    # print("trajlib shape:", trajlib.shape)

    min_idx_h5 = h5py.File(args.idx_file, 'w', driver='mpio', comm=MPI.COMM_WORLD)
    if not "/min_idx" in min_idx_h5:
        min_idx_h5.create_dataset(
            "min_idx",
            shape=(len(ds),),
            dtype='int')
    comm.Barrier()

    chunk_size = math.floor(len(ds) / size)
    lo = chunk_size * rank
    hi = lo + chunk_size if rank != (size - 1) else len(ds)

    iterator = tqdm(range(lo, hi)) if rank==0 else range(lo, hi)
    for i in iterator:
        # 获取第一个样本的数据
        data = ds[i]
        min_idx_h5['min_idx'][i] = find_min_idx_se2(data, trajlib, ds.transer).cpu().numpy()
    min_idx_h5.close()

    del ds

