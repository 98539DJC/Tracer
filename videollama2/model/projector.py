#    Copyright 2024 Alibaba DAMO Academy
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import re

import einops
import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
from timm.models.regnet import RegStage
from timm.models.layers import LayerNorm2d, trunc_normal_
from transformers import TRANSFORMERS_CACHE
import math

def parse_snapshot_folder(repo_id, cache_dir=None, repo_type="model"):
    revision = "main"
    # 1. parse the downloaded cache folder
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    else:
        cache_dir = cache_dir
    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    # 2. resolve refs (for instance to convert main to the associated commit sha)
    refs_dir = os.path.join(repo_cache, "refs")
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()
    # 3. acquire the snapshot folder
    folder = os.path.join(repo_cache, "snapshots", revision)

    return folder


def load_mm_projector(model_path, cache_dir=None, token=None):
    if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
        is_local = True
        folder = model_path
    else:
        is_local = False
        folder = parse_snapshot_folder(model_path, cache_dir=cache_dir, repo_type="model")
        if not os.path.exists(os.path.join(folder, 'mm_projector.bin')):
            # downloading from remote repo
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_path, cache_dir=cache_dir, token=token)

    mm_projector_weights = torch.load(os.path.join(folder, 'mm_projector.bin'), map_location='cpu')
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    return mm_projector_weights


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "linear":
        # NOTE: for both linear and mlp2x_gelu projector type, mean pooling is adopted to aggreate video features
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type == "stc_connector":
        return STCConnector(config)
    elif projector_type == "stp_connector":
        return STPConnector(config)
    elif projector_type == "stc_connector_v35":
        return STCConnectorV35(config)
    elif projector_type == "spatial_conv":
        return SpatialConv(config)
    elif projector_type == "spatial_pool":
        return SpatialPool(config)
    elif projector_type == "tracer" or projector_type == "route_mamba":
        return TracerMambaConnector(config)
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class STCConnector(nn.Module):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        """Temporal Convolutional Vision-Language Connector.
        
        Args:
            config: config object.
            downsample: (temporal, height, width) downsample rate.
            depth: depth of the spatial interaction blocks.
            mlp_depth: depth of the vision-language projector layers.
        """
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size = config.mm_hidden_size
        self.hidden_size = hidden_size = config.hidden_size
        self.output_hidden_size = output_hidden_size = config.hidden_size 
        # TODO: make these as config arguments
        self.depth = depth
        self.mlp_depth = mlp_depth # 2
        self.downsample = downsample # (2, 2, 2)
        if depth != 0:
            self.s1 = RegStage(
                depth=depth, # 4
                in_chs=encoder_hidden_size, # 1152
                out_chs=hidden_size, # 3584
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s1 = nn.Identity()
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=1,
                bias=True
            ),
            nn.SiLU()
        )
        if depth != 0:
            self.s2 = RegStage(
                depth=depth,
                in_chs=hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s2 = nn.Identity()
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)

    def forward(self, x):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        t = x.size(1)
        if x.ndim == 4:
            hw = int(x.size(2) ** 0.5)
            x = einops.rearrange(x, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif x.ndim == 5:
            x = einops.rearrange(x, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        # 1. the first stage of the adapter
        x = self.s1(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)
        # 2. downsampler
        x = self.sampler(x)
        new_t = x.size(2)
        # 3. the second stage of the adapter
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s2(x)
        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        x = self.readout(x)
        return x


class STPConnector(STCConnector):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        self.sampler = nn.Sequential(nn.AvgPool3d(downsample), nn.SiLU())


class STCConnectorV35(STCConnector):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=0,
                bias=True
            ),
            nn.SiLU())


class SpatialConv(STCConnector):

    def __init__(self, config, downsample=(1, 2, 2), depth=0, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)


class SpatialPool(STPConnector):

    def __init__(self, config, downsample=(1, 2, 2), depth=0, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        
class NeuralTrajectoryEvolution(nn.Module):
    """
    NTE:包含连续坐标采样和 ATD 机制
    """
    def __init__(self, d_model, search_radius=5, beta=0.8, tau=0.3):
        super().__init__()
        self.d = d_model
        self.R = search_radius
        self.beta = beta
        self.tau = tau
        
        # Q K V Projections
        self.W_Q = nn.Linear(self.d, self.d)
        self.W_K = nn.Linear(self.d, self.d)
        self.W_V = nn.Linear(self.d, self.d)
        
        # Saccadic Gaze Estimator (输入为 current feat + context) -> \Delta pos
        self.W_pos = nn.Linear(2 * self.d, 2)
        
        # Fidelity Gating Estimator -> sim score
        self.W_sim = nn.Linear(self.d, 1)
        
    def _get_neighborhood_grid(self, pos, H, W):
        """
        基于当前浮点坐标获取局部网格并归一化到 [-1, 1] 供 grid_sample 使用
        pos: (B, N, 2)
        """
        B, N, _ = pos.shape
        device = pos.device
        
        # 生成局部 offset
        offsets_1d = torch.arange(-self.R, self.R + 1, device=device)
        offset_y, offset_x = torch.meshgrid(offsets_1d, offsets_1d, indexing='ij')
        offsets = torch.stack([offset_x.flatten(), offset_y.flatten()], dim=-1) # (K, 2)
        K = offsets.shape[0]
        
        # pos + offset
        pos_expanded = pos.unsqueeze(2) # (B, N, 1, 2)
        offsets_expanded = offsets.unsqueeze(0).unsqueeze(0) # (1, 1, K, 2)
        sample_coords = pos_expanded + offsets_expanded # (B, N, K, 2)
        
        # 归一化到 [-1, 1]
        grid_x = sample_coords[..., 0] / (W - 1) * 2 - 1
        grid_y = sample_coords[..., 1] / (H - 1) * 2 - 1
        normalized_grid = torch.stack([grid_x, grid_y], dim=-1) # (B, N, K, 2)
        
        return normalized_grid

    def _step_evolution(self, x_t, pos_t, c_t, X_v_next):
        """
        x_t: (B, N, D), pos_t: (B, N, 2), c_t: (B, N, 1)
        X_v_next: (B, D, H, W)
        """
        B, D, H, W = X_v_next.shape
        N = x_t.shape[1]
        
        # 1. 局部连续搜索网格采样 (Bilinear)
        grid = self._get_neighborhood_grid(pos_t, H, W) # (B, N, K, 2)
        K = grid.shape[2]
        grid_for_sample = grid.view(B, N, K, 2).to(X_v_next.dtype)
        
        
        # 2. Attention
        q = self.W_Q(x_t).unsqueeze(2) # (B, N, 1, D)
        Xv_flat = X_v_next.permute(0, 2, 3, 1).reshape(-1, self.d)
        K_map = self.W_K(Xv_flat).reshape(B, H,W, self.d).permute(0, 3, 1, 2)
        V_map = self.W_V(Xv_flat).reshape(B, H, W, self.d).permute(0, 3, 1, 2)
        k = F.grid_sample(K_map, grid_for_sample, mode="bilinear",padding_mode="border", align_corners=True).permute(0, 2, 3, 1)
        v = F.grid_sample(V_map, grid_for_sample, mode="bilinear", padding_mode="border",align_corners=True).permute(0, 2, 3, 1)
        
        attn = F.softmax((q @ k.transpose(-1, -2)) / (self.d ** 0.5), dim=-1) # (B, N, 1, K)
        x_t_next_evolved = (attn @ v).squeeze(2) # (B, N, D)
        
        # 3. Saccadic Gaze Estimator (位置更新)
        delta_pos = self.W_pos(torch.cat([x_t_next_evolved, q.squeeze(2)], dim=-1)) # (B, N, 2)
        pos_t_next = pos_t + delta_pos
        
        # 4. Adaptive Temporal Decoupling (ATD)
        sim_score = self.W_sim(x_t_next_evolved * x_t) # Element-wise sim
        c_t_next = torch.sigmoid(self.beta * c_t + (1 - self.beta) * sim_score)
        
        # 产生 Reset Indicator
        i_t_next = (c_t_next < self.tau).to(self.W_Q.weight.dtype)
        
        # 发生断点时原地重采样特征复位
        resample_grid = self._get_neighborhood_grid(pos_t_next, H, W)[:, :, 60, :].unsqueeze(2).to(X_v_next.dtype) # (B, N, 1, 2)
        reset_feat = F.grid_sample(X_v_next, resample_grid, mode='bilinear', align_corners=True)
        reset_feat = reset_feat.permute(0, 2, 3, 1).squeeze(2) # (B, N, D)
        
        # 门控融合
        x_t_next = (1 - i_t_next) * x_t_next_evolved + i_t_next * reset_feat
        
        return x_t_next, pos_t_next, c_t_next, i_t_next

    def forward(self, X_v, reverse=False, pos_init=None):
        """
        X_v: (B, T, D, H, W)
        """
        B, T, D, H, W = X_v.shape
        N = H * W
        device = X_v.device
        X_v = X_v.to(self.W_Q.weight.dtype)
        
        time_steps = list(range(T))
        if reverse:
            time_steps = time_steps[::-1]
            
        x_seq, pos_seq, c_seq, i_seq = [], [], [], []
        
        # 0时刻初始化
        if pos_init is None:
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            pos_t = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).unsqueeze(0).expand(B, N, 2).to(self.W_Q.weight.dtype)
        else:
            pos_t = pos_init
            
        t0 = time_steps[0]
        # 获取初始帧特征
        grid_0 = self._get_neighborhood_grid(pos_t, H, W)[:, :, 60, :].unsqueeze(2).to(X_v[:, t0].dtype)
        x_t = F.grid_sample(X_v[:, t0], grid_0, mode='bilinear', align_corners=True).permute(0, 2, 3, 1).squeeze(2)
        
        c_t = torch.ones(B, N, 1, device=device, dtype=self.W_Q.weight.dtype)
        i_t = torch.zeros(B, N, 1, device=device, dtype=self.W_Q.weight.dtype)
        
        x_seq.append(x_t)
        pos_seq.append(pos_t)
        c_seq.append(c_t)
        i_seq.append(i_t)
        
        # 时序演化
        for t in time_steps[1:]:
            X_v_next = X_v[:, t]
            x_t, pos_t, c_t, i_t = torch.utils.checkpoint.checkpoint(
                self._step_evolution, x_t, pos_t, c_t, X_v_next, use_reentrant=False
            )
            x_seq.append(x_t)
            pos_seq.append(pos_t)
            c_seq.append(c_t)
            i_seq.append(i_t)
            
        if reverse:
            x_seq, pos_seq, c_seq, i_seq = x_seq[::-1], pos_seq[::-1], c_seq[::-1], i_seq[::-1]
            
        x_out = torch.stack(x_seq, dim=2)     # (B, N, T, D)
        pos_out = torch.stack(pos_seq, dim=2) # (B, N, T, 2)
        c_out = torch.stack(c_seq, dim=2)     # (B, N, T, 1)
        i_out = torch.stack(i_seq, dim=2)     # (B, N, T, 1)
        
        return x_out, pos_out, c_out, i_out


class SpatialTrajectoryEvolution(nn.Module):
    """空间轨迹演化 STE：以时间帧为锚点，沿行优先(raster)做全局空间扫描，
    产出空间优先 token，与 NTE 的时间优先 token 对偶。"""
    def __init__(self, d_model):
        super().__init__()
        self.d = d_model
        from mambavision.models.mamba_vision import MambaVisionMixer
        self.norm = nn.LayerNorm(d_model)
        self.scan = MambaVisionMixer(d_model=d_model, d_state=8, d_conv=3, expand=1)

    def forward(self, X_v):                       # (B, T, D, H, W)
        B, T, D, H, W = X_v.shape
        x = X_v.permute(0, 1, 3, 4, 2).reshape(B * T, H * W, D)  # 每帧展平成 729 空间序列
        x = self.scan(self.norm(x)) + x           # 全局空间扫描 + 残差
        return x.reshape(B, T, H * W, D)          # (B, T, N, D)


class TracerMambaConnector(STCConnectorV35):
    """
    整合了 NTE,ATD和BCM的端到端 Tracer。
    不计算 Loss，而是将所需的所有追踪状态作为 loss_data 输出。
    """
    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2, 
                 mv_stages=1, mv_depth=6, window_size=13, num_heads=32, 
                 mlp_ratio=4.0, drop_rate=0.0, attn_drop=0.0, drop_path=0.1, 
                 layer_scale=None, use_attention_every=2):
                 
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        dim = self.hidden_size
        
        # 1. 挂载 NTE (Neural Trajectory Evolution)
        self.nte = NeuralTrajectoryEvolution(d_model=dim)
        self.nte.apply(self._init_weights)
        # 位置/置信度预测层：W_pos 用小方差让初始位移为 sub-pixel（避免漂移出图，又不落入"不动"平凡解使 loss_cyc 恒 0）；W_sim 零初始化让 fidelity 初始稳定
        nn.init.normal_(self.nte.W_pos.weight, std=1e-3)
        nn.init.zeros_(self.nte.W_pos.bias)
        nn.init.zeros_(self.nte.W_sim.weight)
        nn.init.zeros_(self.nte.W_sim.bias)

        # 1.5 挂载 STE (Spatial Trajectory Evolution，空间轨迹演化)
        self.ste = SpatialTrajectoryEvolution(d_model=dim)
        self.ste.apply(self._init_weights)

        # 2. 构建BCM的 MambaVisionLayer 网络
        from mambavision.models.mamba_vision import BiMambaVisionLayer
        total_blocks = max(1, mv_stages * mv_depth)
        dpr_all = [drop_path * i / (total_blocks - 1) if total_blocks > 1 else 0.0 for i in range(total_blocks)]

        def _attn_positions(mv_depth, every: int):
            if every and every > 0:
                return list(range(every - 1, mv_depth, every))
            return []
            
        layers = []
        cursor = 0
        for _ in range(mv_stages):
            dpr_stage = dpr_all[cursor: cursor + mv_depth] if drop_path > 0 else 0.0
            cursor += mv_depth
            layers.append(
                BiMambaVisionLayer(
                    dim=dim, depth=mv_depth, num_heads=num_heads, window_size=window_size,
                    conv=False, downsample=False, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                    drop=drop_rate, attn_drop=attn_drop, drop_path=dpr_stage, layer_scale=layer_scale,
                    layer_scale_conv=None, transformer_blocks=_attn_positions(mv_depth, use_attention_every),
                )
            )
            
        self.m1 = nn.Sequential(*layers)
        self.m1.apply(self._init_weights)
        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        """
        返回: (Output_Features, Loss_Data_Dict)
        """
        t = x.size(1)
        if x.ndim == 4:
            hw = int(x.size(2) ** 0.5)
            x = einops.rearrange(x, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif x.ndim == 5:
            x = einops.rearrange(x, "b t h w d -> b d t h w")

        # 先用 s1 (RegStage) 把视觉特征维度 (mm_hidden_size) 映射到 LLM hidden size
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s1(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)

        B,D,T,H,W = x.shape
        X_v = einops.rearrange(x, "b d t h w -> b t d h w")
        
        # (1) Neural Trajectory Evolution (前向)
        x_fwd, pos_fwd, c_fwd, i_fwd = self.nte(X_v, reverse=False)

        # (1.5) Spatial Trajectory Evolution (空间轨迹，与时间轨迹对偶)
        x_spatial = torch.utils.checkpoint.checkpoint(self.ste, X_v, use_reentrant=False)
        x_fwd = x_fwd + x_spatial.transpose(1, 2)   # (B, N, T, D) 相加融合

        # (2) Neural Trajectory Evolution (后向循环，用于 Cycle-Consistency)
        pos_init_bwd = pos_fwd[:, :, -1, :].detach() 
        _, pos_bwd, _, _ = self.nte(X_v, reverse=True, pos_init=pos_init_bwd)
        
        # 打包计算 Loss 必需的中间数据
        loss_data = {
            'x_fwd': x_fwd,       # (B, N, T, D) 用于 Semantic Consistency (L_cor)
            'pos_fwd': pos_fwd,   # (B, N, T, 2) 用于 Trajectory Diversity (L_div)
            'c_fwd': c_fwd,       # (B, N, T, 1) Fidelity scores 加权
            'i_fwd': i_fwd,       # (B, N, T, 1) Reset Indicators 截断
            'pos_bwd': pos_bwd    # (B, N, T, 2) 用于 Cycle-Consistency (L_cyc)
        }
        
        # (3) BCM 建模 (MambaVision)
        # BiMambaVisionLayer 期望 5D 输入 (b, t, d, h, w)
        # 注意：此处 x_fwd 的 N 维度本质上是展平的 H*W
        x_mamba_in = einops.rearrange(x_fwd, "b (h w) t d -> b t d h w", h=H, w=W)
        x_mamba_out = torch.utils.checkpoint.checkpoint(self.m1, x_mamba_in, use_reentrant=False)

        # (4) 展平成 [B, N_seq, D] 进入 LLM Projector
        x_out = einops.rearrange(x_mamba_out, "b t d h w -> b (t h w) d")
        x_out = self.readout(x_out)
        
        return x_out, loss_data