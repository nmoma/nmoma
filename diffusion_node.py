'''
This code piece implements a ROS node that provides DDPM planning service.
#! The theta of car chassis seems to be bugged, not sure why
# TODO : implement configuration compliant behavior
# HACK : currently use start point as the odom, might not be the best choice
'''
import torch
import time
# import matplotlib.pyplot as plt

import rospy
import sensor_msgs.point_cloud2 as pc2
from einops import rearrange, repeat

from utils.data_loader import MomaTrajTrans
from test_ddpm import *
from planner.srv import Plan, PlanRequest, PlanResponse

model = None
transer = None
wps_num = None
state_dim = None
trajlib = None
hot_data = None

def handle_plan_request(req : PlanRequest):
    global hot_data
    start_time = time.time()

    sample_num = req.path_num

    data = {}
    # fill data with start and goal
    req.start = torch.tensor(req.start)
    start = torch.zeros(11)
    start[:2]   = 0
    start[2]    = torch.cos(req.start[2])
    start[3]    = torch.sin(req.start[2])
    start[4:11] = req.start[3:]

    req.goal = torch.tensor(req.goal)
    goal = torch.zeros(11)
    goal[:2]   = req.goal[:2] - req.start[:2]
    goal[2]    = torch.cos(req.goal[2])
    goal[3]    = torch.sin(req.goal[2])
    goal[4:11] = req.goal[3:]
    start_goal = torch.cat([start, goal]).reshape(1,2,11)

    wps, tup = norm_trajs(start_goal)
    startgoal = torch.cat([wps[:, 0, -7:], tup, wps[:, -1, -9:]], dim=1)
    # data['startgoal'] = startgoal.expand(sample_num, -1)
    data['startgoal'] = startgoal.expand(1, -1)

    # fill data with point cloud
    point_list = []
    for point in pc2.read_points(req.input_cloud, skip_nans=True):
        point_list.append([point[0], point[1], point[2]])
    point_cloud = torch.tensor(point_list)
    point_cloud = point_cloud.reshape(-1, 3)
    offset = torch.zeros(3)
    offset[:2] = req.start[:2]
    point_cloud -= offset

    c = torch.cos(req.start[2])
    s = torch.sin(req.start[2])
    R = torch.tensor([
        [c, -s, 0], 
        [s, c, 0],
        [0, 0, 1]])
    point_cloud = torch.matmul(point_cloud, R)
    point_cloud = rotate_points(point_cloud, startgoal.squeeze(0))

    # filter those out of [-5, 5] bound
    indices = torch.logical_and(point_cloud[:, 0] > -5, point_cloud[:, 0] < 5)
    indices = torch.logical_and(indices, point_cloud[:, 1] > -5)
    indices = torch.logical_and(indices, point_cloud[:, 1] < 5)
    point_cloud = point_cloud[indices]

    size_ = point_cloud.shape[0]
    nums = 4096
    data['pos'] = torch.zeros(1, nums, 3)
    # data['pos'] = torch.rand(1, nums, 3)
    # data['pos'][:, :, :2] *= 20.0
    # data['pos'][:, :, :2] -= 10.0
    # data['pos'][:, :, 2] *= 1.6
    if size_ > 0:
        if size_ < nums:
            # fill in the point and leave the rest as zeros
            rospy.logwarn("Leaving the rest as zeros")
            data['pos'][0, :size_, :] = point_cloud
        else: 
            # sample 4096 points randomly
            rospy.logwarn("Sampling 4096 out of {} points randomly".format(size_))
            data['pos'][0] = point_cloud[torch.randperm(size_)[:nums], :]
    
    # visualization of pointcloud to validate the transformation
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='r', label='Input Point Cloud')
    # ax.set_aspect('equal', 'box')
    # ax.legend()

    # def on_key_event(event):
    #     if event.key == 'q':
    #         plt.close()
    #         sys.exit()
    # plt.gcf().canvas.mpl_connect('key_press_event', on_key_event)
    # plt.show()

    count = 0
    offset = []
    for item in data['pos']:
        count += item.shape[1]
        offset.append(count)
    offset = torch.IntTensor(offset).expand(1)
    # offset = torch.IntTensor(offset).expand(sample_num)

    data['offset'] = offset
    data['pos'] = data['pos'].expand(1, -1, -1)
    # data['pos'] = data['pos'].expand(sample_num, -1, -1)

    #! value of data['x'] does nothing, consider refactoring
    #! first and last waypoints are reset to data['x'][:, 0, :] and data['x'][:, -1, :]
    #! does not make much sense
    data['x'] = torch.zeros(1, wps_num, state_dim)
    data['x'][0, 0, :] = start
    data['x'][0,-1, :] = goal
    data['x'] = data['x'].expand(sample_num, -1, -1)
    #! data['feat'] does nothing
    data['feat'] = None

    req.boxes = torch.tensor(req.boxes).reshape(-1, 8)
    req.boxes[..., :2] -= req.start[:2]
    indices = torch.logical_and(req.boxes[:, 0] > -5, req.boxes[:, 0] < 5)
    indices = torch.logical_and(indices, req.boxes[:, 1] > -5)
    indices = torch.logical_and(indices, req.boxes[:, 1] < 5)
    req.boxes = req.boxes[indices]

    data['boxes'] = torch.zeros((50, 8))
    num_boxes = min(req.boxes.shape[0], 50)
    data['boxes'][:num_boxes, :] = req.boxes[:num_boxes]
    data['boxes'] = data['boxes'].reshape(1, 50, 8)
    data['boxes'] = repeat(data['boxes'], '1 n d -> k n d', k=1)
    data['mu'] = transer.mu.unsqueeze(0).repeat(sample_num, 1, 1)
    data['std'] = transer.std.unsqueeze(0).repeat(sample_num, 1, 1)

    for d in data.keys():
        if data[d] is not None:
            # data[d] = data[d].cuda()
            data[d] = data[d].cuda().half()

    # if hot_data is None:
    #     hot_data = data.copy()

    torch.cuda.synchronize()
    t0 =time.perf_counter()
    # outputs, predict_prims, prbs = model.sample(trajlib, data)
    outputs, predict_prims, prbs = model.sample(trajlib.half(), data)
    torch.cuda.synchronize()
    t1 =time.perf_counter()
    _sample_time = (t1-t0) * 1000

    print("Time :", _sample_time)

    outputs = outputs[:, 0, -1].float()
    outputs = transer.detrans((outputs, data['startgoal'][:, 7:10].float()))

    # outputs = outputs[0].cpu()
    outputs = outputs.cpu()
    outputs = torch.cat([outputs[..., :2], torch.atan2(outputs[..., [3]], outputs[..., [2]]), outputs[..., 4:]], dim=-1)

    # outputs[:, :2] += req.start[:2]
    outputs[:,  0, :] = req.start
    outputs[:, -1, :] = req.goal

    _success = True
    _path_length = outputs.shape[1]
    _path = list(outputs.flatten().detach().numpy())
    _message = "Received DDPM Plan Request"
    # return PlanResponse(False, 0, [], "Received DDPM Plan Request")
    return PlanResponse(_success, _path_length, _path, _sample_time, _message)

def get_hot(_):
    global hot_data

    if hot_data is not None:
        if 'cond' in hot_data:
            hot_data.pop('cond')
        # outputs, predict_prims, prbs = model.sample(trajlib, hot_data)
        outputs, predict_prims, prbs = model.sample(trajlib.half(), hot_data)

def planner_server():
    rospy.init_node('ddpm_planner_server')
    s = rospy.Service('ddim_plan', Plan, handle_plan_request)
    # rospy.Timer(rospy.Duration(0.1), get_hot)
    rospy.loginfo("DDPM Planner Server is up and running")
    rospy.spin()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ddpm.yaml', help='Path to config file')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='Device to use')
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)

    # if cfg.diffuser.trunc_type == "gt":
    #     cfg.diffuser.trunc_type = "trunc"

    # dataset = MomaDataset("data_train_raw.h5", 
    #                       cfg.data_normer, 
    #                       mustd_path="logs/mustd.npy", )
    # wps_num = dataset.wps_num
    # state_dim = dataset.state_dim
    
    trajlib = np.load("logs/trajlib32.npy", allow_pickle=True)
    trajlib = torch.as_tensor(trajlib).float().contiguous().to(device)

    wps_num = 64
    state_dim = 11
    mu, std = np.load("logs/mustd_nonfix.npy")
    transer = MomaTrajTrans(norm_method=cfg.data_normer, mu=torch.tensor(mu), std=torch.tensor(std))

    unet = UNetModel(cfg, slurm=False).to(device)
    model = DDPM(transer, unet, cfg.diffuser).to(device)
    model.eval()
    print("Loading checkpoint from {0}".format(cfg.ckpt_dir))
    load_ckpt(model, path=cfg.ckpt_dir + '/model.pth', map_location=device)
    model = model.half()

    hot_data = None

    # start = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # goal =  (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # input_cloud = PointCloud2()
    # req = PlanRequest(start, goal, input_cloud)
    # response = handle_plan_request(req)
    # print(response.message)

    planner_server()
