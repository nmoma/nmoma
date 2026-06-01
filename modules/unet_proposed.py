import os
from typing import Dict
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import SPACER, timestep_embedding, filter
from utils.utils import SGOccFusion, ResBlock, SpatialTransformer, BasicTransformerBlock, PositionEmbeddingLearned
from omegaconf import DictConfig, OmegaConf
from modules.mink_resnet import MinkResNet
from modules.ptrans3 import Ptrans3
from modules.pt_mamba import PointMamba
import MinkowskiEngine as ME

class ResFully(nn.Module):
    def __init__(self, n_model, n_layer=2, use_layer_norm=True):
        super(ResFully, self).__init__()
        layers = []
        for i in range(n_layer):
            layers.append(nn.Linear(n_model, n_model))
            layers.append(nn.LayerNorm(n_model) if use_layer_norm else nn.Identity())
            if i != n_layer-1:
                layers.append(nn.SiLU())
        self.fnet = nn.Sequential(*layers)
        
    def forward(self, x):
        return nn.SiLU()(self.fnet(x) + x)

class UNetModel(nn.Module):
    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:
        super(UNetModel, self).__init__()

        self.d_x = cfg.d_x
        self.d_model = cfg.d_model
        self.d_startgoal = cfg.d_startgoal
        self.nblocks = cfg.nblocks
        self.resblock_dropout = cfg.resblock_dropout
        self.transformer_num_heads = cfg.transformer_num_heads
        self.transformer_dim_head = cfg.transformer_dim_head
        self.transformer_dropout = cfg.transformer_dropout
        self.transformer_depth = cfg.transformer_depth
        self.transformer_mult_ff = cfg.transformer_mult_ff
        self.context_dim = self.d_model + self.d_startgoal # for transformer input
        self.use_position_embedding = cfg.use_position_embedding # for input sequence x
        self.d_sgin = 0
        if cfg.data_normer == "simple":
            self.d_sgin = 7+11 
        elif cfg.data_normer == "shoot": 
            self.d_sgin = 7+3+9 
        self.spacer = SPACER[cfg.data_normer]

        ## create scene model from config
        self.cond_backbone = cfg.scene_model.cond_backbone
        self.add_spacer = cfg.scene_model.add_spacer
        if self.cond_backbone == "simple":
            blocks = [8, 16, 64, 256, 512, 512, 512, 512]
            if self.add_spacer:
                blocks[0] += 3
            res_layers = 2
            res_dim = 512
            use_layer_norm = True
            use_final_norm = True
            add_transformer = False
            layers = []
            for i in range(len(blocks) - 1):
                layers.append(nn.Linear(blocks[i], blocks[i+1]))
                layers.append(nn.LayerNorm(blocks[i+1]) if use_layer_norm else nn.Identity())
                layers.append(nn.SiLU())
                if blocks[i+1] == blocks[i] and blocks[i+1] == res_dim:
                    layers.append(ResFully(res_dim, res_layers, use_layer_norm))
            layers.append(nn.Linear(blocks[-1], self.d_model))
            layers.append(nn.LayerNorm(self.d_model) if use_final_norm else nn.Identity())
            
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, 
                                                        nhead=4, 
                                                        dim_feedforward=128, 
                                                        dropout=0.0, activation=nn.SiLU())
            if add_transformer:
                layers.append(nn.TransformerEncoder(encoder_layer, num_layers=6))
            self.scene_model = nn.Sequential(*layers)
            self.sg_fusion = BasicTransformerBlock(dim=self.d_model+self.d_startgoal, n_heads=2, d_head=128, 
                                                   dropout=0.0, context_dim=self.d_startgoal, 
                                                   gated_ff=True, mult_ff=2)
        else:
            cfg.scene_model.c = 3
            self.voxel_size = 0.05
            self.context_dim = self.d_model
            self.pos_embed_env = PositionEmbeddingLearned(3, self.d_model)
            blocks = [4, 16, 64, 256]
            layers =[]
            for i in range(len(blocks) - 1):
                layers.append(nn.Linear(blocks[i], blocks[i+1]))
                layers.append(nn.LayerNorm(blocks[i+1]))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(blocks[-1], self.d_model))
            layers.append(nn.LayerNorm(self.d_model))
            self.sg_encoder = nn.Sequential(*layers)
            self.sg_fusion = SGOccFusion(dim=self.d_model, n_heads=4, d_head=64, 
                                        dropout=0.0, gated_ff=True, mult_ff=2)
            if self.cond_backbone == "minkresnet":
                self.scene_model = MinkResNet(depth=18, in_channels=cfg.scene_model.c, max_channels=self.d_model)
                # self.scene_model = MinkResNet(depth=34, in_channels=cfg.scene_model.c, max_channels=self.d_model)
            elif self.cond_backbone == "ptans3":
                self.scene_model = Ptrans3(in_channels=cfg.scene_model.c,
                                           enc_channels=(32, 64, 128, 256, 256),
                                           enc_num_head=(2, 4, 8, 16, 16))
            elif self.cond_backbone == "ptmamba":
                self.scene_model = PointMamba(config=cfg.scene_model.ptmamba)
            else:
                raise ValueError("Invalid cond_backbone")
        
        time_embed_dim = self.d_model * cfg.time_embed_mult
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.in_layers = nn.Sequential(
            nn.Conv1d(self.d_x, self.d_model, 1)
        )

        self.layers = nn.ModuleList()
        for i in range(self.nblocks):
            self.layers.append(
                ResBlock(
                    self.d_model,
                    time_embed_dim,
                    self.resblock_dropout,
                    self.d_model,
                )
            )
            self.layers.append(
                SpatialTransformer(
                    self.d_model, 
                    self.transformer_num_heads, 
                    self.transformer_dim_head, 
                    depth=self.transformer_depth,
                    dropout=self.transformer_dropout,
                    mult_ff=self.transformer_mult_ff,
                    context_dim=self.context_dim,
                )
            )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, 1),
        )
        
        self.start_goal_enc = None
        if self.d_startgoal > 0:
            self.start_goal_enc = nn.Sequential(
                nn.Linear(self.d_sgin, 32),
                nn.SiLU(),
                nn.Linear(32, 64),
                nn.SiLU(),
                nn.Linear(64, self.d_startgoal)
            )
        
        # classify
        self.cls_num = 32
        self.cls_pool = nn.AdaptiveMaxPool1d(1)  # 全局平均池化
        self.cls_out = nn.Sequential(
            nn.Linear(self.context_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, self.cls_num),
        )
        self.cls_model = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.d_model),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 1),
        )
        
    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """ Apply the model to an input batch

        Args:
            x_t: the input data, <B, C> or <B, L, C>
            ts: timestep, 1-D batch of timesteps
            cond: condition feature
        
        Return:
            the denoised target data, i.e., $x_{t-1}$
        """
        in_shape = len(x_t.shape)
        if in_shape == 2:
            x_t = x_t.unsqueeze(1)
        assert len(x_t.shape) == 3

        ## time embedding
        t_emb = timestep_embedding(ts, self.d_model)
        t_emb = self.time_embed(t_emb)

        h = rearrange(x_t, 'b l c -> b c l')
        h = self.in_layers(h) # <B, d_model, L>
        # print(h.shape, cond.shape) # <B, d_model, L>, <B, T , c_dim>

        ## prepare position embedding for input x
        if self.use_position_embedding:
            B, DX, TX = h.shape
            pos_Q = torch.arange(TX, dtype=h.dtype, device=h.device)
            pos_embedding_Q = timestep_embedding(pos_Q, DX) # <L, d_model>
            h = h + pos_embedding_Q.permute(1, 0) # <B, d_model, L>

        for i in range(self.nblocks):
            h = self.layers[i * 2 + 0](h, t_emb)
            h = self.layers[i * 2 + 1](h, context=cond)
        
        # anchor classify
        anchor_prob = None
        if B // self.cls_num > 2:
            h4cls = rearrange(self.cls_pool(h).squeeze(-1), '(a b) c -> a b c', a=self.cls_num)
            anchor_prob = self.cls_model(h4cls).squeeze(-1)
        
        h = self.out_layers(h)
        h = rearrange(h, 'b c l -> b l c')

        ## reverse to original shape
        if in_shape == 2:
            h = h.squeeze(1)

        # h = filter(h)
        
        return h, anchor_prob

    def condition(self, data: Dict) -> torch.Tensor:

        b = data['startgoal'].shape[0]
        
        if self.cond_backbone == "simple":
            startgoals = data['startgoal']
            env_in = data["boxes"]
            if self.add_spacer:
                feat_occ = torch.Tensor([0., 0., 1.]).cuda().repeat(b, env_in.shape[1], 1)
                env_in = torch.cat([env_in, feat_occ], dim=2)
                psg = self.spacer(startgoals)
                env_in = torch.cat([env_in, psg], dim=1) # b x (50+18) x (8+3)
            cond_feat = self.scene_model(env_in)
            if self.start_goal_enc is not None:
                startgoal_emd = self.start_goal_enc(startgoals).unsqueeze(1)
                startgoal_emd_full = startgoal_emd.repeat(1, cond_feat.shape[1], 1)
                cond_feat = torch.cat([cond_feat, startgoal_emd_full], dim=2)
                cond_feat = self.sg_fusion(cond_feat, context=startgoal_emd)
        elif self.cond_backbone == "minkresnet":
            pos, startgoals = data['pos'], data['startgoal']
            env_in = pos.clone()
            coordinates, features = ME.utils.batch_sparse_collate(
                    [(p[:, :3] / self.voxel_size, p[:, 0:] if p.shape[1] > 3 else p[:, :3]) for p in env_in],
                    device=env_in[0].device) 
            env_sp = ME.SparseTensor(coordinates=coordinates, features=features)
            env_sp = self.scene_model(env_sp)
            # get info of sparse tensor
            sampled_coords,sampled_features = [],[]
            len_x = []
            for permutation in env_sp.decomposition_permutations:
                len_x.append(len(env_sp.coordinates[permutation]))
            max_len_x = int(torch.tensor(len_x).max())
            if len(len_x)>1:
                for permutation in env_sp.decomposition_permutations:
                    if len(permutation) > max_len_x:
                        choice = torch.randperm(len(permutation))[:max_len_x]
                        choice = torch.sort(choice).values
                        sampled_features.append(env_sp.features[permutation][choice])
                        sampled_coords.append(env_sp.coordinates[permutation][choice])
                    else:
                        padding_size = max_len_x - len(permutation)      
                        padded_features = torch.cat(
                            [env_sp.features[permutation], torch.zeros((padding_size, env_sp.features[permutation].shape[1]), 
                                                                dtype=env_sp.features.dtype).to(env_sp.device)], dim=0) 
                        padded_coords = torch.cat(
                            [env_sp.coordinates[permutation], -torch.ones((padding_size, env_sp.coordinates[permutation].shape[1]),
                                                                    dtype=env_sp.coordinates.dtype).to(env_sp.device)], 
                                                                    dim=0)   
                        sampled_features.append(padded_features)
                        sampled_coords.append(padded_coords)
            else:
                for permutation in env_sp.decomposition_permutations:
                    sampled_features.append(env_sp.features[permutation])
                    sampled_coords.append(env_sp.coordinates[permutation])
            sampled_features = torch.stack(sampled_features).contiguous()
            sampled_coords = torch.stack(sampled_coords)
            pos_feats = self.pos_embed_env(sampled_coords[:,:,1:]*self.voxel_size).transpose(1, 2).contiguous()
            sampled_features = sampled_features + pos_feats
            # sg fusion
            psg = self.sg_encoder(self.spacer(startgoals))
            psg = rearrange(psg, 'b l c -> b c l')
            pos_Q = torch.arange(psg.shape[-1], dtype=psg.dtype, device=psg.device)
            pos_embedding_Q = timestep_embedding(pos_Q, psg.shape[-2]) # <L, d_model>
            psg = psg + pos_embedding_Q.permute(1, 0) # <B, d_model, L>
            psg = rearrange(psg, 'b c l -> b l c')
            cond_feat = self.sg_fusion(psg, sampled_features, occ_mask=(sampled_coords[:, :, -1] == -1))
        elif self.cond_backbone == "ptans3":
            pos, startgoals = data['pos'], data['startgoal']
            env_in = pos.clone()
            offset = torch.ones(b, dtype=torch.int).cuda() * env_in.shape[1]
            for i in range(b):
                offset[i] *= (i+1)
            points_in = {'coord': rearrange(pos, 'b n c -> (b n) c'),
                        'grid_size': self.voxel_size,
                        'feat': rearrange(env_in, 'b n c -> (b n) c'),
                        'offset': offset
            }
            points_out = self.scene_model(points_in)
            poses = []
            feats = []
            max_len = max(points_out['offset'].diff().max(), points_out['offset'][0]).item()
            for i in range(b):
                if i==0:
                    begin = 0
                else:
                    begin = points_out['offset'][i-1]
                end = points_out['offset'][i]
                coords = points_out['coord'][begin:end]
                coords_f = points_out['feat'][begin:end]
                if max_len > end-begin:
                    padding_size = max_len - (end-begin)
                    coords_f = torch.cat(
                        [coords_f, torch.zeros((padding_size, coords_f.shape[1]), 
                                                dtype=coords_f.dtype).to(coords_f.device)], dim=0) 
                    coords = torch.cat(
                        [coords, -torch.ones((padding_size, coords.shape[1]), 
                                                dtype=coords.dtype).to(coords.device)], dim=0) 
                poses.append(coords)
                feats.append(coords_f)
            poses = torch.stack(poses).contiguous()
            feats = torch.stack(feats)
            pos_feats = self.pos_embed_env(poses*self.voxel_size).transpose(1, 2).contiguous()
            feats = feats + pos_feats
            # sg fusion
            psg = self.sg_encoder(self.spacer(startgoals))
            psg = rearrange(psg, 'b l c -> b c l')
            pos_Q = torch.arange(psg.shape[-1], dtype=psg.dtype, device=psg.device)
            pos_embedding_Q = timestep_embedding(pos_Q, psg.shape[-2]) # <L, d_model>
            psg = psg + pos_embedding_Q.permute(1, 0) # <B, d_model, L>
            psg = rearrange(psg, 'b c l -> b l c')
            cond_feat = self.sg_fusion(psg, feats, occ_mask=(poses[:, :, -1] == -1))
        elif self.cond_backbone == "ptmamba":
            pos, startgoals = data['pos'], data['startgoal']
            pts, feats = self.scene_model(pos, pos)
            pos_feats = self.pos_embed_env(pts*self.voxel_size).transpose(1, 2).contiguous()
            feats = feats + pos_feats
            # sg fusion
            psg = self.sg_encoder(self.spacer(startgoals))
            psg = rearrange(psg, 'b l c -> b c l')
            pos_Q = torch.arange(psg.shape[-1], dtype=psg.dtype, device=psg.device)
            pos_embedding_Q = timestep_embedding(pos_Q, psg.shape[-2]) # <L, d_model>
            psg = psg + pos_embedding_Q.permute(1, 0) # <B, d_model, L>
            psg = rearrange(psg, 'b c l -> b l c')
            cond_feat = self.sg_fusion(psg, feats, occ_mask=None)
        return cond_feat
    
    def classify(self, cond: torch.Tensor):
        if cond.dim() > 2:
            x = cond.permute(0, 2, 1)
            x = self.cls_pool(x).squeeze(-1)
        else:
            x = cond
        output = self.cls_out(x)
        return output.view(-1, self.cls_num)

if __name__ == '__main__':
    cfg = OmegaConf.load("config/ddpm.yaml")
    unet = UNetModel(cfg, slurm=False).cuda()
    batch_size = 8
    point_num = cfg.scene_model.num_points
    startgoal = torch.rand(batch_size, unet.d_sgin).cuda()
    pos = torch.rand(batch_size, point_num, 3).cuda() * 10.0
    cond_data = {'pos': pos,
                'startgoal': startgoal}

    x_t = torch.rand(batch_size, 16, 11).cuda()
    tim = torch.randint(0, 10, (batch_size,)).cuda()
    cond = unet.condition(cond_data)
    print(cond.shape)
    h = unet(x_t, tim, cond)
    print(h.shape)
    cls_tensor = unet.classify(cond)
    print(cls_tensor.shape)

# if __name__ == '__main__':
#     cfg = OmegaConf.load("config/ddpm.yaml")
#     unet = UNetModel(cfg, slurm=False)
#     batch_size = 2
#     point_num = cfg.scene_model.num_points
#     startgoal = torch.rand(batch_size, unet.d_sgin)
#     pos = torch.rand(point_num * batch_size, 3)
#     boxes = torch.rand(batch_size, 50, 8)
#     cond_data = {'pos': pos,
#                 'startgoal': startgoal,
#                 'boxes': boxes}

#     x_t = torch.rand(batch_size, 16, 11)
#     tim = torch.randint(0, 10, (batch_size,))
#     cond = unet.condition(cond_data)
#     print(cond.shape)
#     h = unet(x_t, tim, cond)
#     print(h.shape)
#     cls_tensor = unet.classify(cond)
#     print(cls_tensor.shape)
    