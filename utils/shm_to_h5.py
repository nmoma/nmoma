# This file is used to convert the data into hdf5 format.
# It uses MPI and parallel HDF5 to parallelize the process.
#! h5py must be compiled with parallel support to use this feature. (see https://docs.h5py.org/en/latest/mpi.html#parallel)
#! hdf5 1.14.5 with python 3.8 causes "not a datatype problem" (see https://github.com/h5py/h5py/issues/2436)
#! use hdf 1.12.3 with python 3.8 to avoid this problem.

import sys
from os import listdir
from os.path import isfile, join

from tqdm import tqdm
import numpy as np
import h5py
from mpi4py import MPI

from shared_data import SharedDataConsumer

wps_num = 64
state_dim = 11
occ_pointnum = 4096

def farthest_point_sampling(points, num_samples):
    """
    NumPy-optimized Farthest Point Sampling
    points: numpy array of shape (N, 3)
    num_samples: number of points to sample
    """
    N = points.shape[0]

    if N <= num_samples:
        end_time = time.time()
        print(f"NumPy FPS sampling: {end_time - start_time:.4f}s (no sampling needed)")
        return points

    # Initialize
    selected_indices = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(N, np.inf, dtype=np.float64)

    # Start with a random point
    selected_indices[0] = np.random.randint(0, N)

    # Precompute squared norms for efficient distance calculation
    points_sq = np.sum(points ** 2, axis=1)

    for i in range(1, num_samples):
        # Get last selected point
        last_idx = selected_indices[i-1]
        last_point = points[last_idx]
        last_sq = points_sq[last_idx]

        # Compute squared distances using: ||x - y||² = ||x||² + ||y||² - 2x·y
        # This avoids expensive square root operations until necessary
        dot_product = np.dot(points, last_point)
        sq_distances = points_sq + last_sq - 2 * dot_product

        # Update minimum distances (element-wise min)
        np.minimum(distances, sq_distances, out=distances)

        # Select point with maximum minimum distance
        selected_indices[i] = np.argmax(distances)

    result = points[selected_indices]

    return result

def idx2map3d(idx_coords):
    map_coords = (idx_coords + 0.5) * 0.1 + np.array([-8.0, -8.0, 0.0])
    return map_coords

def main(rank : int, world_size : int, total_size : int, output_file : str, shm_name : str):
    f = h5py.File(output_file, 'w', driver='mpio', comm=MPI.COMM_WORLD)
    dataset = {}
    dataset['wps'] = f.create_dataset("wps",
        shape=(total_size, wps_num, state_dim),
        dtype='float64')

    dataset['occs'] = f.create_dataset("occs",
        shape=(total_size, occ_pointnum, 3),
        dtype='float16')

    dataset['sdf'] = f.create_dataset("sdf",
        shape=(total_size, 160, 160, 16),
        dtype='float16')

    dataset['boxes'] = f.create_dataset("boxes",
        shape=(total_size, 50, 8),
        dtype='float32')

    dataset['min_idx'] = f.create_dataset('min_idx',
            shape=(total_size,),
            dtype='float16')

    consumer = SharedDataConsumer(shm_name)
    datapoint_num = int(total_size / world_size)
    if rank == 0:
        datapoint_num += args.number % world_size
    offset = rank * datapoint_num


    iterator = range(datapoint_num)
    if rank == 0:
        iterator = tqdm(iterator)
    for i in tqdm(range(datapoint_num)):
        datapoint = consumer.consume()
        idx = offset + i

        # ===wps===
        wps = np.array(datapoint.traj, dtype='float32')
        theta = wps[:, [2]]
        wps = np.concatenate(
            (wps[:, :2], np.cos(theta), np.sin(theta), wps[:, 3:]), axis=1
        )
        dataset['wps'][idx] = wps

        # ===sdf3d===
        sdf_3d = np.array(datapoint.ESDF3D, dtype='float32').reshape(160, 160, 16)
        dataset['sdf'][idx] = sdf_3d.astype(np.float16)

        # ===boxes===
        # dp_boxes = np.array(datapoint.boxes, dtype='float32').reshape(-1, 8)
        # # fill with zero if not enough or truncate if exceeded
        # if dp_boxes.shape[0] < 50:
        #     fill_num = 50 - dp_boxes.shape[0]
        #     dp_boxes = np.concatenate([dp_boxes, np.zeros((fill_num, 8), dtype=np.float32)], axis=0)
        # elif dp_boxes.shape[0] > 50:
        #     dp_boxes = dp_boxes[:50, :]
        # dataset['boxes'][idx] = dp_boxes

        # ===occs===
        sdf_3d[:1] = 10.0
        sdf_3d[-1:] = 10.0
        sdf_3d[:, :1] = 10.0
        sdf_3d[:, -1:] = 10.0

        sdf_points_idx = np.argwhere(np.abs(sdf_3d)<0.01)
        sdf_points_world = None
        if sdf_points_idx.shape[0] < occ_pointnum:
            sdf_points_idx = np.pad(sdf_points_idx, ((0, occ_pointnum-sdf_points_idx.shape[0]), (0,0)))
            sdf_points_world = idx2map3d(sdf_points_idx)
        elif sdf_points_idx.shape[0] > occ_pointnum:
            # indices = np.random.choice(sdf_points_idx.shape[0], occ_pointnum, replace=False)
            # sdf_points_idx = sdf_points_idx[indices]
            sdf_points_world = idx2map3d(sdf_points_idx)
            sdf_points_world = farthest_point_sampling(sdf_points_world, occ_pointnum)
        dataset['occs'][idx] = sdf_points_world.astype(np.float16)
    f.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', type=int)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-s', '--shm-name', type=str)
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size

    main(rank, size, args.number, args.output, args.shm_name)
    sys.exit(0)
