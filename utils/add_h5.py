import h5py
import os
from tqdm import tqdm
import time

def get_dataset_shapes(input_folder):
    # file_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.h5')]
    file_paths = [os.path.join(input_folder, file) for file in ["data.h5", "data1.h5", "data2.h5", "data3.h5", "data4.h5"]]
    dataset_shapes = {}
    print(file_paths)
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as input_file:
            for dataset_name, dataset in input_file.items():
                if dataset_name not in dataset_shapes:
                    dataset_shapes[dataset_name] = dataset.shape
                else:
                    dataset_shapes[dataset_name] = (dataset_shapes[dataset_name][0] + dataset.shape[0],) + dataset_shapes[dataset_name][1:]

    return dataset_shapes

input_folder = '/media/xulong/fa8391eb-583e-4acc-aa15-c0e4d91be259/xulong/nmoma/src/planner/log/train-1'
origin_file_path = '/media/xulong/fa8391eb-583e-4acc-aa15-c0e4d91be259/xulong/nmoma/src/planner/log/train-1/data.h5'
dataset_shapes = get_dataset_shapes(input_folder)
print(dataset_shapes)
origin_idx = 0
# file_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.h5')]
file_paths = [os.path.join(input_folder, file) for file in ["data.h5", "data1.h5", "data2.h5", "data3.h5", "data4.h5"]]
with h5py.File(origin_file_path, 'r+') as f:
    for dataset_name, dataset_shape in dataset_shapes.items():
        print('datasetnam: ',dataset_name)
        print('dataset_shape: ',dataset_shape)
        origin_idx = f[dataset_name].shape[0]
        f[dataset_name].resize(dataset_shape)
    for file_path in file_paths:
        if file_path == origin_file_path:
            print("origin continue.")
            continue
        print(file_path)
        cur_idx = origin_idx
        with h5py.File(file_path, 'r') as input_file:
            for dataset_name in input_file:
                print(dataset_name)
                cur_idx = origin_idx
                for i in tqdm(range(input_file[dataset_name].shape[0])):
                    f[dataset_name][cur_idx:cur_idx+1] = input_file[dataset_name][i]
                    cur_idx += 1
            print(cur_idx)
        origin_idx = cur_idx
