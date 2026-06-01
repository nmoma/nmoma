from utils.data_loader import MomaDataset, collate_fn_4ptrans
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from utils.utils import *
from sklearn.cluster import KMeans
import random
from torch.utils.data import Dataset, DataLoader

batch_size = 1000
ds = MomaDataset(filepath="data_valid_raw.h5", 
                 norm_method="shoot")
train_dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn_4ptrans,
        # num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last = True,
        shuffle=True
    )

trajs = None
for it, data in tqdm(enumerate(train_dataloader)):
    traj = data['x']
    traj = traj * ds.std + ds.mu
    if trajs is None:
        trajs = traj
    else:
        trajs = torch.cat([trajs, traj], dim=0)
trajs = trajs.numpy()
print(trajs.shape)
    
primitive_num = 31
# primitive_num = 1023

# for i in range(trajs.shape[0]):
#     wps = torch.as_tensor(trajs[i]).unsqueeze(0)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     points_wps_cuda, points_wps_idx = get_robo_tensor(wps.cuda())
#     # 绘制 wps
#     points_wps = points_wps_cuda.cpu().numpy()[0]
#     for j in range(points_wps.shape[0]):
#         color = plt.cm.coolwarm(j / points_wps.shape[0])  # 使用冷暖渐变颜色
#         ax.plot(points_wps[j, :, 0], points_wps[j, :, 1], points_wps[j, :, 2], c=color, marker='o')
                            
#     ax.set_aspect('equal', 'box')
#     manager = plt.get_current_fig_manager()
#     manager.resize(2048, 1660)
#     def on_key_event(event):
#         if event.key == 'q':
#             plt.close()
#             sys.exit()
#     plt.gcf().canvas.mpl_connect('key_press_event', on_key_event)
#     plt.show()
   
#! K-Means
kmeans = KMeans(n_clusters=primitive_num)
print("begin k-means ...")
kmeans.fit(trajs.reshape(-1, trajs.shape[1]*trajs.shape[2]))
print("end k-means ...")
centers = np.array(kmeans.cluster_centers_).reshape(-1, trajs.shape[1], trajs.shape[2])

traj0 = np.zeros((1, trajs.shape[1], trajs.shape[2]))
for i in range(traj0.shape[1]):
    traj0[0][i][0] = 1.0 * i / (traj0.shape[1]-1)
    traj0[0, i, 2] = 1.0
centers = np.append(traj0, centers, axis=0)

print(centers.shape)
np.save('logs/trajlib32.npy', centers)

# wps = torch.as_tensor(centers).float().cuda()
# wps[..., -7:] *= torch.as_tensor(max_joint).float().cuda()
# points_wps_cuda, points_wps_idx = get_robo_tensor(wps)
# points_wps = points_wps_cuda.cpu().numpy()

#! full state
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in tqdm(range(points_wps.shape[0])):
#     color = (random.random(), random.random(), random.random())
#     for j in range(points_wps.shape[1]):
#         ax.plot(points_wps[i, j, :, 0], points_wps[i, j, :, 1], points_wps[i, j, :, 2], c=color, marker='o')
                        
# ax.set_aspect('equal', 'box')
# manager = plt.get_current_fig_manager()
# manager.resize(2048, 1660)
# def on_key_event(event):
#     if event.key == 'q':
#         plt.close()
#         sys.exit()
# plt.gcf().canvas.mpl_connect('key_press_event', on_key_event)
# plt.show()

#! ee pose
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in tqdm(range(points_wps.shape[0])):
#     ax.plot(points_wps[i, :, -1, 0], points_wps[i,:, -1, 1], points_wps[i, :, -1, 2], 'r-')
# ax.set_xlabel('x', fontsize=15)
# ax.set_ylabel('y', fontsize=15)
# ax.set_aspect('equal', 'box')
# plt.show()

#! chassis
fig, ax =  plt.subplots()
for j in tqdm(range(centers.shape[0])):
    traj = centers[j]
    ax.plot(traj[:, 0], traj[:, 1], 'r-')
    for i in range(traj.shape[0]):
        dir = np.array([traj[i, 2], traj[i, 3]]) / 10.0
        ax.arrow(traj[i, 0], traj[i, 1],
                dir[0], dir[1],
                head_width=0.01, head_length=0.02, fc='blue', ec='blue')
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)
ax.set_aspect('equal', 'box')
plt.show()
