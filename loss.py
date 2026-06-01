from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.utils import *

# map_size_3d is the numbder of cells in each dimension of the 3D map
# map_size is the size of the 3D map in meters
map_size = [d * map_resolution for d in map_size_3d]

def positiveSmoothedL1(x):

    #x: B, ...
    pe = 1.0e-4
    half = 0.5 * pe
    f3c = 1.0 / (pe * pe)
    f4c = -0.5 * f3c / pe

    b1 = x <= 0.0
    b2 = (x>0.0) & (x < pe)
    b3 = x >= pe

    a1 = 0.0
    a2 = (f4c * x + f3c) * x * x * x
    a3 = x - half
    loss = a1 * b1 + a2 * b2 + a3 * b3

    return loss

def positiveSmoothedL2(x):
    #x:B,...
    f = nn.ReLU()
    x = f(x)
    loss  = x * x
    return loss

def positiveSmoothedExp(x):
    #x:B,...
    f = nn.ReLU()
    x = f(x)
    loss = torch.exp(x) - 1
    return loss

def positiveSmoothedL3(x):
    #x:B,...
    f = nn.ReLU()
    x = f(x)
    loss  = x * x *x
    return loss

def posToIndex(pos, res, ori):
    # pos: B*3 ori: B*3 res: double
    if pos.dim() == 2:
        tmppos = pos - ori
    elif pos.dim() == 3:
        tmppos = pos - ori.unsqueeze(1)
    elif pos.dim() == 4:
        tmppos = pos - ori.unsqueeze(1).unsqueeze(2)
    else:
        raise ValueError("Unsupported pos dimensions: {}".format(pos.dim()))
    gridIdx = ((tmppos / res).floor()).long()
    return gridIdx

def indexToPos(grid, res, ori):

    pos = Variable(((grid+0.5)*res)) #B*C*3 or B*3

    if(pos.dim()==2):
        return pos + ori
    elif pos.dim()==3:
        return pos + ori.unsqueeze(dim=1)
    elif pos.dim() == 4:
        return pos + ori.unsqueeze(1).unsqueeze(2)
    else:
        raise ValueError("Unsupported grid dimensions: {}".format(grid.dim()))

def indexToAddress(grid):
    #[1, 1, 1]
    #[C, C, C]
    # cols = int(map_size[1]/map_resolution)
    # heis = int(map_size[2]/map_resolution)
    cols = map_size_3d[1]
    heis = map_size_3d[2]
    if len(grid) == 3:
        return grid[0]*cols*heis + grid[1]*heis + grid[2]
    elif len(grid) == 4:
        return grid[:, 0] * cols * heis + grid[:, 1] * heis + grid[:, 2]
    raise ValueError("Unsupported grid dimensions: {}".format(grid.dim()))

def boundIdx(idx):
    # B*c*3
    rows = int(map_size[0]/map_resolution)
    cols = int(map_size[1]/map_resolution)
    heis = int(map_size[2]/map_resolution)
    if idx.dim() == 3:
        idx[:,:,0] = torch.clamp(idx[:,:,0], 0, rows-1)
        idx[:,:,1] = torch.clamp(idx[:,:,1], 0, cols-1)
        idx[:,:,2] = torch.clamp(idx[:,:,2], 0, heis-1)
    elif idx.dim() == 2:
        idx[:,0] = torch.clamp(idx[:,0], 0, rows-1)
        idx[:,1] = torch.clamp(idx[:,1], 0, cols-1)
        idx[:,2] = torch.clamp(idx[:,2], 0, heis-1)
    else:
        raise ValueError("Unsupported idx dimensions: {}".format(idx.dim()))
    return idx

def getDistanceWithTrilinear(esdfmap, pos, param_origin):
    # esdfmap: B*(X_size*Y_size*Z_size)
    # pos: B*3
    # origin: B*3
    param_resolution = map_resolution
    pos_temp = torch.zeros_like(pos)
    pos_temp[:, 0] = torch.clamp(pos[:, 0], param_origin[:,0]+param_resolution, param_origin[:,0] + map_size[0]-param_resolution)
    pos_temp[:, 1] = torch.clamp(pos[:, 1], param_origin[:,1]+param_resolution, param_origin[:,1] + map_size[1]-param_resolution)
    pos_temp[:, 2] = torch.clamp(pos[:, 2], param_origin[:,2]+param_resolution, param_origin[:,2] + map_size[2]-param_resolution)
    pos_m = pos_temp - 0.5 * param_resolution
    idx = posToIndex(pos_m,param_resolution,param_origin )
    idx_pos = indexToPos(idx, param_resolution, param_origin)#zhchzc
    diff = (pos_temp - idx_pos) / param_resolution #B*3

    Bs = pos.shape[0]
    values = torch.zeros(Bs, 2, 2, 2).to(pos.device)
    for x in range(0, 2):
        for y in range(0, 2):
            for z in range(0, 2):
                offset = Variable(torch.tensor([x,y,z]).long().to(pos.device))
                current_idx = idx + offset #B*3
                current_idx = boundIdx(current_idx)
                if current_idx.dim() == 2:
                    idx_ = torch.cat([
                        torch.arange(0, Bs).reshape(-1,1).to(pos.device),
                        indexToAddress([current_idx[:,0], current_idx[:,1],current_idx[:,2]]).reshape(-1,1)], 
                    dim=1)
                    values[:,x,y,z] = esdfmap[idx_[:, 0], idx_[:, 1]]
                else:
                    pass
                    raise Exception("Unsupported current_idx dimensions: {}".format(current_idx.dim()))
                    # values[i,x,y,z] = esdfmap[i,indexToAddress([current_idx[i,:,0], current_idx[i,:,1],current_idx[i,:,2]])]

    v00 = (1 - diff[:,0]) * values[:,0,0,0] + diff[:,0] * values[:,1,0,0]
    v01 = (1 - diff[:,0]) * values[:,0,0,1] + diff[:,0] * values[:,1,0,1]
    v10 = (1 - diff[:,0]) * values[:,0,1,0] + diff[:,0] * values[:,1,1,0]
    v11 = (1 - diff[:,0]) * values[:,0,1,1] + diff[:,0] * values[:,1,1,1]
    v0 = (1 - diff[:,1]) * v00 + diff[:,1] * v10
    v1 = (1 - diff[:,1]) * v01 + diff[:,1] * v11
    dist = (1 - diff[:,2]) * v0 + diff[:,2] * v1
    return dist

def getDistanceFromState(moma_state : torch.Tensor, sdf : torch.Tensor) -> torch.Tensor:  
    """get sdf value of collision points given the moma state

    Args:
        moma_state (torch.Tensor): N x 11 states of moma
        sdf (torch.Tensor): M x (MAP_X*MAP_Y*MAP_Z) sdf values of the map\
            M=N or M=1; different maps are used for different state respectively in the former case; \
            the same map is used for all states in the latter case.

    Returns:
        torch.Tensor: (N x 13) sdf values of collision points, 13 is the number of collision points per state
    """    
    dev = moma_state.device
    N = moma_state.shape[0]
    # return getDistanceWithTrilinear(
    #     esdfmap = torch.repeat_interleave(sdf, 13, dim=0) if sdf.shape[0] == N else sdf.expand(13*N, -1), # (N x 13) x (MAP_X*MAP_Y*MAP_Z)
    #     pos = getColliPts(moma_state).view(-1, 3), # (N x 13) x 3
    #     param_origin = torch.tensor([-5., -5., 0.], device=dev).expand(13*N, -1) # (N x 13) x 3
    # ).view(N, 13)

    # when only one sdf is provided, expand can avoid memory allocation
    if sdf.shape[0] == 1:
        return getDistanceWithTrilinear(
            esdfmap = sdf.expand(13*N, -1), # (N x 13) x (MAP_X*MAP_Y*MAP_Z)
            pos = getColliPts(moma_state).view(-1, 3), # (N x 13) x 3
            param_origin = torch.tensor([-5., -5., 0.], device=dev).expand(13*N, -1) # (N x 13) x 3
        ).view(N, 13)

    # when one sdf is provided for each state, iterate over number of collision point to avoid memory allocation
    assert sdf.shape[0] == N, "sdf shape should be 1 or N"
    colliPts = getColliPts(moma_state) # (N x 13 x 3)
    distances = \
    [
        getDistanceWithTrilinear(
            esdfmap = sdf, # N x (MAP_X*MAP_Y*MAP_Z)
            pos = colliPts[:, i, :], # N x 3
            param_origin = torch.tensor([-5., -5., 0.], device=dev).expand(N, -1) # N x 3
        ) 
        for i in range(13)
    ]
    return rearrange(distances, "D N -> N D")

def selfDistanceMatrix(colli_pts : torch.Tensor) -> torch.Tensor:
    """Compute the distance between each pair of collision points, returns a N x 13 x 13 tensor
        Colliding points are marked with negative values, whose absolute values are the penetration depths.
        Collision points that are not allowed to collide, as specified by moma_param.h, are marked as inf

    Args:
        colli_pts (torch.Tensor): N x 13 x 3 tensor of collision points

    Returns:
        distance_matrix : N x 13 x 13 tensor of distance, see above for details
    """       
    radius = torch.tensor(getColliPtsRadius(), dtype=colli_pts.dtype, device=colli_pts.device)
    dist_matrix = torch.full((colli_pts.shape[0], 13, 13), torch.inf,
                                    dtype=colli_pts.dtype, device=colli_pts.device)

    colliMat = getColliMatrix()
    # deal with chassis-arm collision separately since, the chassis is modeled as a cylinder
    #   compute cylinder-sphere distance
    dy = colli_pts[:, 3:, 2] - MOMA_PARAM.chassis_height

    dx = torch.norm(colli_pts[:, 3:, :2] - colli_pts[:, 0, :2].unsqueeze(1), dim=2)
    dx = dx.clone()

    dx-= MOMA_PARAM.chassis_colli_radius
    dx = dx.clone()

    sqr_dist = torch.zeros_like(dx, dtype=colli_pts.dtype, device=colli_pts.device)
    sqr_dist = sqr_dist.clone()
    yidx = dy > 0.0
    sqr_dist[yidx] = dy[yidx] ** 2
    
    sqr_dist = sqr_dist.clone()
    xidx = dx > 0.0
    sqr_dist[xidx]+= dx[xidx] ** 2

    dist = torch.sqrt(sqr_dist) - radius[3:]
    dist = dist.clone()
    inside_idx = torch.logical_and(torch.logical_not(xidx), torch.logical_not(yidx)) # for those lies inside the chassis
    dist[inside_idx] = torch.max(dy[inside_idx], dx[inside_idx])


    dist_matrix[:, 0, 3:] = dist
    dist_matrix[:, 3:, 0] = dist

    # deal with arm-arm collision
    for i in range(1, 12):
        for j in range(i+1, 13):
            if not colliMat[i-1,j-1]: continue
            dist = dist.clone()
            dist = torch.norm(colli_pts[:,i,:] - colli_pts[:,j,:], dim=1)
            dist_matrix[:, i,j] = dist - radius[i] - radius[j]
            # dist_matrix[:, j,i] = dist_matrix[:, i,j] 

    return dist_matrix

def isStateFree(moma_state : torch.Tensor, sdf : torch.Tensor, safe_margin : float = 1e-3) -> torch.Tensor:
    """Tells whether the given state is free or not, given the sdf map and safe margin \
    moma_collision 

    Args:
        moma_state (torch.Tensor): N x 11 states of moma
        sdf (torch.Tensor): M x (MAP_X*MAP_Y*MAP_Z) sdf values of the map\
            M=N or M=1; different maps are used for different state respectively in the former case; \
            the same map is used for all states in the latter case.

    Returns:
        is_free: 1D tensor of N elements of boolean indicating whether the state is free or not
    """    
    N, dev = moma_state.shape[0], moma_state.device
    assert (sdf.shape[0] == 1 or sdf.shape[0] == N), "sdf shape should be 1 or N"
    is_free = torch.full((moma_state.shape[0],), True, device=dev)

    # check for moma limits
    is_free &= torch.all(moma_state[:, -7:] <= torch.tensor(MOMA_PARAM.getJointLimitsMax(), device=moma_state.device).expand(N, -1), dim=1)
    is_free &= torch.all(moma_state[:, -7:] >= torch.tensor(MOMA_PARAM.getJointLimitsMin(), device=moma_state.device).expand(N, -1), dim=1)

    # check for moma-moma collision
    colli_pts = getColliPts(moma_state)
    dist_matrix = selfDistanceMatrix(colli_pts).flatten(start_dim=1) # N x (13 x 13)
    
    is_free &= (dist_matrix > safe_margin).all(dim=1)

    # check moma-obstacle collision
    distance = getDistanceWithTrilinear(
        esdfmap = sdf.expand(13*N, -1) if sdf.shape[0] == 1 else torch.repeat_interleave(sdf, 13, dim=0), # (N x 13) x (MAP_X*MAP_Y*MAP_Z)
        pos = colli_pts.view(-1, 3), # (N x 13) x 3
        param_origin = torch.tensor([-5., -5., 0.], device=dev).expand(13*N, -1) # (N x 13) x 3
    ).view(N, 13)
    distance -= torch.tensor(getColliPtsRadius(), device=moma_state.device)
    is_free &= (distance > safe_margin).all(dim=1)
    # print(torch.all(is_free))

    # check moma-ground collision
    radius = torch.tensor(getColliPtsRadius(), device=dev)
    arm_heights = colli_pts[:, 1:, 2] # N x 12
    is_free &= torch.all(arm_heights - radius[1:].expand(N, -1) > safe_margin, dim=1)
    
    return is_free

def diffSO2(theta1, theta2):
    # theta1: Bx2 (cos, sin)
    # theta2: Bx2 (cos, sin)
    cd = theta1[..., 0]*theta2[..., 0] + theta1[..., 1]*theta2[..., 1]
    sd = theta1[..., 1]*theta2[..., 0] - theta1[..., 0]*theta2[..., 1]
    return torch.atan2(sd, cd).abs()

# TODO: add turn loss
def TimeLoss(traj_in):
    #inputs: B*wps_num*11
    #return: B
    pts1 = traj_in[:,:-1,:] 
    pts2 = traj_in[:,1:,:]
    diff = pts1 - pts2 #B*wps_num-1*11
    dist_so2 = diffSO2(pts1[..., 2:4], pts2[..., 2:4]) #B*wps_num-1
    dist_r2 = diff[..., :2].norm(dim=2) # B*wps_num-1
    mask = dist_r2 > 1e-2
    
    dist_so2[mask] = diffSO2(pts1[..., 2:4][mask], diff[..., :2][mask]) + \
                     diffSO2(pts2[..., 2:4][mask], diff[..., :2][mask])
                             
    time = dist_r2 / max_v + dist_so2 / max_w #B*wps_num-1
    jvl = torch.as_tensor(joint_vel_limit).to(traj_in.device).unsqueeze(0).unsqueeze(1) \
        .repeat(traj_in.shape[0], traj_in.shape[1]-1, 1)
    time = torch.maximum(time, (diff[..., -7:].abs() / jvl).max(-1)[0])
    return time.sum(-1)

def UniTimeLoss(traj_in, gt):
    #inputs: B*wps_num*11
    #return: 1
    pts1 = traj_in[:,:-1,:] 
    pts2 = traj_in[:,1:,:]
    diff = pts1 - pts2 #B*wps_num-1*11
    dist_so2 = diffSO2(pts1[..., 2:4], pts2[..., 2:4]) #B*wps_num-1
    dist_r2 = diff[..., :2].norm(dim=2) # B*wps_num-1
    mask = dist_r2 > 1e-2
    
    dist_so2[mask] = diffSO2(pts1[..., 2:4][mask], diff[..., :2][mask]) + \
                     diffSO2(pts2[..., 2:4][mask], diff[..., :2][mask])
                             
    time = dist_r2 / max_v + dist_so2 / max_w #B*wps_num-1
    
    jvl = torch.as_tensor(joint_vel_limit).to(traj_in.device).unsqueeze(0).unsqueeze(1) \
        .repeat(traj_in.shape[0], traj_in.shape[1]-1, 1)
    time = torch.maximum(time, (diff[..., -7:].abs() / jvl).max(-1)[0])
    varloss = torch.std(time, dim=-1, unbiased=False)
    normalizeLoss = varloss / TimeLoss(gt)
    loss = torch.mean(normalizeLoss)
    return loss

def ArcLoss(cors):
    #inputs: B*100*2
    pts1 = cors[:,:-1,:2]
    pts2 = cors[:,1:,:2]
    arcs = pts1 - pts2 #B*99*3
    arcs = torch.norm(arcs, dim=2)#B*99
    loss = torch.sum(arcs, dim=1)#B
    return loss

def UniArcLoss(inputs, gt, reduction="mean"):
    pts1 = inputs[:,:-1,:2]
    pts2 = inputs[:,1:,:2]
    arcs = pts1 - pts2 #B*99*3
    arcs = torch.norm(arcs, dim=2)#B*99
    varloss = torch.std(arcs, dim=1, unbiased=False) #B
    labelArc = ArcLoss(gt) #B
    normalizeLoss = varloss / labelArc
    if reduction == "mean":
        normalizeLoss = torch.mean(normalizeLoss)
    return normalizeLoss

def SmoothLoss(traj_in):
    #inputs: B*wps_num*11
    #return: 1
    pts1 = traj_in[:,:-1,:2] 
    pts2 = traj_in[:,1:,:2]
    v = pts2-pts1
    normV = torch.nn.functional.normalize(v, dim=2)#B 99 2
    v1 = normV[:,:-1,:]
    v2 = normV[:,1: ,:]
    c = torch.zeros_like(v2)#B 98 2
    c[:,:,0] = -v2[:,:,1]
    c[:,:,1] =  v2[:,:,0]
    cross = torch.sum(v1 * c,dim=2)#B*98
    cross = torch.pow(cross,2)#b 98
    loss = torch.sum(cross, dim=1)#B
    return loss

def ThetaSmoothLoss(traj_in):
    #inputs: B*wps_num*11
    #return: 1
    pts1 = traj_in[:,:-1,2:4] 
    pts2 = traj_in[:,1:,2:4]
    v = pts2-pts1
    normV = torch.nn.functional.normalize(v, dim=2)#B 99 2
    cross = torch.sum(torch.pow(normV, 2), dim=2)#b 99
    loss = torch.sum(cross, dim=1)#B
    return loss

def QSmoothLoss(traj_in):
    #inputs: B*wps_num*11
    #return: 1
    pts1 = traj_in[:,:-1, 4:]
    pts2 = traj_in[:,1:, 4:]
    v = pts2-pts1
    normV = torch.nn.functional.normalize(v, dim=2)#B 99 2
    cross = torch.sum(torch.abs(normV), dim=2)#b 99
    loss = torch.sum(cross, dim=1)#B
    return loss

def NormedSmoothLoss(inputs, gt, reduction="mean"):
    loss1 = SmoothLoss(inputs)
    loss2 = SmoothLoss(gt)
    vios = loss1-loss2
    sloss = positiveSmoothedL1(vios) #B
    if reduction == "mean":
        sloss = torch.mean(sloss)
    return sloss

def ThetaNormedSmoothLoss(inputs, gt, reduction="mean"):
    loss1 = ThetaSmoothLoss(inputs)
    loss2 = ThetaSmoothLoss(gt)
    vios = loss1-loss2
    sloss = positiveSmoothedL1(vios) #B
    if reduction == "mean":
        sloss = torch.mean(sloss)
    return sloss

def QNormedSmoothLoss(inputs, gt, reduction="mean"):
    loss1 = QSmoothLoss(inputs)
    loss2 = QSmoothLoss(gt)
    vios = loss1-loss2
    sloss = positiveSmoothedL1(vios) #B
    if reduction == "mean":
        sloss = torch.mean(sloss)
    return sloss

def UniEEArcLoss(inputs, gt, reduction="mean"):
    B, W, _ = inputs.shape
    pts = getColliPts(inputs.reshape(-1, 11)).reshape(B, -1, 3) # B x (13 x W) x 3
    pts = pts[:, 12::13, :] # B x W x 3
    arcs = pts[:,:-1,:] - pts[:,1:,:] # B x (W-1) x 3
    arcs = torch.norm(arcs, dim=2) # B x (W-1)
    varloss = torch.std(arcs, dim=1, unbiased=False) # B
    if reduction == "mean":
        varloss = torch.mean(varloss)
    return varloss

# def UniEEArcLoss(inputs, gt, reduction="mean"):
#     return ThetaNormedSmoothLoss(inputs, gt, reduction) + QNormedSmoothLoss(inputs, gt, reduction)

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, output, target):
        # output: (batch_size, num_classes)
        # target: (batch_size,)
        
        pt = F.softmax(output, dim=1)
        log_pt = torch.log(pt)
        
        target_index = target.view(-1, 1).long()
        pt = pt.gather(1, target_index)
        log_pt = log_pt.gather(1, target_index)
        loss = -torch.mul(torch.pow((1 - pt), self.gamma), log_pt)
        alpha =  self.alpha.to(output.device).gather(0, target.view(-1))
        loss = torch.mul(alpha, loss.t())

        return loss.mean()

class AnchorLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(AnchorLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred, target):
        """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

        Args:
            pred (torch.Tensor): The prediction with shape (N, C), C is the
                number of classes
            target (torch.Tensor): The learning label of the prediction.
        """
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        # Actually, pt here denotes (1 - pt) in the Focal Loss paper
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        # Thus it's pt.pow(gamma) rather than (1 - pt).pow(gamma)
        focal_weight = (self.alpha * target + (1 - self.alpha) *
                        (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight
        
        return loss.mean()

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    esdf = torch.zeros(1, 100, 100, 16)
    for i in range(16):
        esdf[0, i, i, i] = i
    esdf = esdf.flatten(start_dim=1)
    
    # params_origin = torch.tensor([0,0,0], dtype=torch.float32).reshape(1,-1)
    # pos = torch.ones(1, 3) * 54
    # pos.requires_grad = True
    # dist = getDistanceWithTrilinear(esdf, pos, params_origin)

    # dist.backward()
    # print(pos.grad)

    dist =getDistanceFromState(
        moma_state = torch.zeros(1,11),
        sdf = torch.ones(1, map_size_3d[0] * map_size_3d[1] * map_size_3d[2])
    )
    print(dist)

    # The following code piece is for testing the collision checking function
    # while True:
    #     N = 1
    #     sdf = torch.ones(1, map_size_3d[0] * map_size_3d[1] * map_size_3d[2]) * 100
    #     moma_state = torch.randn((N, 11), dtype=float)
        
    #     moma_state[:, 2] = 1.0
    #     moma_state[:, 3] = 0.0

    #     moma_state[:, -7:] = torch.clamp(moma_state[:, -7:], 
    #         min=torch.tensor(MOMA_PARAM.getJointLimitsMin()), 
    #         max=torch.tensor(MOMA_PARAM.getJointLimitsMax()))
    #     colli_pts = getColliPts(moma_state) # N x 13 x 3
    #     colli_pts.requires_grad = True
    #     dist_matrix = selfDistanceMatrix(colli_pts)[0]
    #     objective = torch.sum(dist_matrix[dist_matrix < 0])
    #     objective.backward()
    #     grad_ = colli_pts.grad.clone()

    #     grad_ = grad_.detach().numpy()
    #     grad_ = grad_.reshape(-1, 3)
    #     colli_pts = colli_pts.detach().numpy()
    #     free_ = isStateFree(moma_state, sdf, safe_margin=0)
    #     idx = torch.argwhere(dist_matrix < 0)
    #     colli_ = []
    #     if idx.shape[0] > 0:
    #         colli_ = list(zip(*idx.tolist()))[0]

    #     pts = colli_pts.reshape(-1, 3)

    #     radius = utils.getColliPtsRadius()

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     for i in range(N):
    #         joint = 0
    #         # Create the data points for the cylinder
    #         num_points = 20
    #         theta = np.linspace(0, 2 * np.pi, num_points)
    #         z = np.linspace(0, MOMA_PARAM.chassis_height, num_points)
    #         theta_grid, z_grid = np.meshgrid(theta, z)

    #         # Convert theta and z to x and y, using the equation of a circle
    #         x_grid = MOMA_PARAM.chassis_colli_radius * np.cos(theta_grid) + np.array(colli_pts[i, joint, 0])
    #         y_grid = MOMA_PARAM.chassis_colli_radius * np.sin(theta_grid) + np.array(colli_pts[i, joint, 1])
    #         color = 'red' if joint in colli_ else 'gray'
    #         ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.2)
    #         for joint in range(1, 13):
    #             theta = np.linspace(0, 2 * np.pi, 10)
    #             phi = np.linspace(0, np.pi, 10)
    #             theta, phi = np.meshgrid(theta, phi)
    #             r = radius[joint] 
    #             x = r * np.sin(phi) * np.cos(theta) + np.array(colli_pts[i, joint, 0])
    #             y = r * np.sin(phi) * np.sin(theta) + np.array(colli_pts[i, joint, 1])
    #             z = r * np.cos(phi) + np.array(colli_pts[i, joint, 2])
    #             color = 'red' if joint in colli_ else 'gray'
    #             ax.plot_surface(x, y, z, color=color, alpha=0.2)
    #             ax.quiver(pts[:, 0], pts[:, 1], pts[:, 2], grad_[:, 0], grad_[:, 1], grad_[:, 2], length=0.05, arrow_length_ratio=0.5, normalize=True)

    #     ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='red', marker='o')

    #     ax.set_aspect('equal', 'box')
    #     manager = plt.get_current_fig_manager()
    #     manager.resize(2048, 1660)

    #     def on_key_event(event):
    #         if event.key == 'q':
    #             plt.close()
    #             exit()
    #     plt.gcf().canvas.mpl_connect('key_press_event', on_key_event)
    #     plt.show()