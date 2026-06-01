# This file is used to convert the data into hdf5 format.
# It uses MPI and parallel HDF5 to parallelize the process.
#! h5py must be compiled with parallel support to use this feature. (see https://docs.h5py.org/en/latest/mpi.html#parallel)
#! hdf5 1.14.5 with python 3.8 causes "not a datatype problem" (see https://github.com/h5py/h5py/issues/2436)
#! use hdf 1.12.3 with python 3.8 to avoid this problem.

import torch
import numpy as np
import sys
from utils.utils import idx2map3d

from os import listdir
from os.path import isfile, join

import h5py
import time

from data import DataLoader  # type: ignore

wps_num = 64
state_dim = 11
occ_pointnum = 4096
mode = "train"
# mode = "simple"

def main():
    data_dir = "src/planner/log/" + mode
    file_list = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
    file_list.sort()
    dataloader_list = [DataLoader(f) for f in file_list]
    size_list = [dl.size() for dl in dataloader_list]
    total_size = sum(size_list)

    output_file = "data_" + mode + "_raw.h5"
    f = h5py.File(output_file, 'w')
    wps = f.create_dataset("wps",
        shape=(total_size, wps_num, state_dim),
        # maxshape=(None, wps_num, state_dim),
        # chunks=(chunk_size, wps_num, state_dim),
        dtype='float64')

    occs = f.create_dataset("occs",
        shape=(total_size, occ_pointnum, 3),
        # maxshape=(None, occ_pointnum, 3),
        # chunks=(chunk_size, occ_pointnum, 3),
        dtype='float16')
    
    sdf = f.create_dataset("sdf",
        shape=(total_size, 100, 100, 16),
        # maxshape=(None, 100, 100, 16),
        # chunks=(chunk_size, 100, 100, 16),
        dtype='float16')
    
    boxes = f.create_dataset("boxes",
        shape=(total_size, 50, 8),
        dtype='float32')

    min_idx = f.create_dataset('min_idx', 
            shape=(total_size,), 
            # maxshape=(None,),
            # chunks=(chunk_size,), 
            dtype='float16')

    buffer_size = 64
    wps_buffer = []
    occs_buffer = []
    sdf_buffer = []
    boxes_buffer = []

    vis_cnt = 0
    for i in range(len(file_list)):
        offset = sum(size_list[:i])
        dataset_idx = offset
        file_name = file_list[i]
        print("Processing file: ", file_name)
        dl = DataLoader(file_name)
        while dl.hasNext():
            while dl.hasNext() and len(wps_buffer) < buffer_size:
                try:
                    datapoint = dl.next()
                except:
                    raise Exception("Error in reading data from file: ", file_name)
                
                vis_cnt += 1
                print("Processing data point: ", vis_cnt)
                
                wps_raw = torch.as_tensor(np.array(datapoint.traj, dtype='float64'))
                wps_raw = torch.cat([wps_raw[:, :2], torch.cos(wps_raw[:, 2]).unsqueeze(1), 
                                    torch.sin(wps_raw[:, 2]).unsqueeze(1), wps_raw[:, 3:]], dim=1)
                wps_buffer.append(wps_raw.squeeze(0).cpu().numpy())
                
                sdf_3d = np.array(datapoint.ESDF3D, dtype='float64').reshape(100, 100, 16)
                sdf_buffer.append(sdf_3d.astype(np.float16))

                dp_boxes = np.array(datapoint.boxes, dtype='float32').reshape(-1, 8)
                # fill with zero if not enough or truncate if exceeded
                if dp_boxes.shape[0] < 50:
                    fill_num = 50 - dp_boxes.shape[0]
                    dp_boxes = np.concatenate([dp_boxes, np.zeros((fill_num, 8), dtype=np.float32)], axis=0)
                elif dp_boxes.shape[0] > 50:
                    dp_boxes = dp_boxes[:50, :]
                boxes_buffer.append(dp_boxes)
                
                # sdf_3d = sdf_3d[1:-1, 1:-1]
                sdf_3d[:1] = 10.0
                sdf_3d[-1:] = 10.0
                sdf_3d[:, :1] = 10.0
                sdf_3d[:, -1:] = 10.0
                sdf_points_idx = np.argwhere(np.abs(sdf_3d)<0.01)
                sdf_points_idx = torch.tensor(sdf_points_idx).contiguous()
                if sdf_points_idx.shape[0] < occ_pointnum:
                    addded = torch.zeros(occ_pointnum-sdf_points_idx.shape[0], 3, dtype=torch.int)
                    sdf_points_idx = torch.cat([sdf_points_idx, addded], dim=0)
                else:
                    indices = torch.randperm(sdf_points_idx.size(0))[:occ_pointnum]
                    sdf_points_idx = sdf_points_idx[indices]
                sdf_points_world = idx2map3d(sdf_points_idx)
                occs_buffer.append(sdf_points_world.cpu().numpy().astype(np.float16))
            wps[dataset_idx:dataset_idx+len(wps_buffer)] = np.stack(wps_buffer)
            occs[dataset_idx:dataset_idx+len(wps_buffer)] = np.stack(occs_buffer)
            sdf[dataset_idx:dataset_idx+len(wps_buffer)] = np.stack(sdf_buffer)
            boxes[dataset_idx:dataset_idx+len(wps_buffer)] = np.stack(boxes_buffer)
            min_idx[dataset_idx:dataset_idx+len(wps_buffer)] = np.zeros(len(wps_buffer), dtype=np.float16)

            dataset_idx += len(wps_buffer)
            wps_buffer = []
            occs_buffer = []
            sdf_buffer = []
            boxes_buffer = []
    f.close()

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Finished.")
    print("Time elapsed: ", time.time() - start_time, "seconds")
    sys.exit(0)
