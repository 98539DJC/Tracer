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
import copy

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
    elif projector_type == "vim_spatial":
        return VimSpatialConnector(config)
    elif projector_type == "vim_spatial_s1":
        # Stage1：空间单分支（1352 token、无 NTE 辅助损失）
        return VimSpatialConnector(config, spatial_only=True)
    elif projector_type == "vim_spatial_fuse":
        # 阶段 B：时间查空间的交叉注意力融合（固定 1352 token，方案 v1 §4）
        return VimSpatialConnector(config, fusion=True)
    elif projector_type == "vim_spatial_joint":
        # 联合训练：时间/空间/融合/LoRA 同一阶段一起训，两路保留计算图
        return VimSpatialConnector(config, fusion=True, joint=True)
    elif projector_type == "tracer_vim_cross":
        # 方案 v2：Tracer 时间分支 + 逐帧 Vim + Cross-Mamba + RFM
        # （st_fusion_mode/st_scan_mode/st_align_mode/st_fusion_scale/
        #   st_rank_weight/st_frame_chunk 由 config 传入）
        return SpatioTemporalCrossConnector(config)
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
    def __init__(self, d_model, search_radius=5, beta=0.8, tau=0.5):
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

        # ---- v13: residual attention -------------------------------------
        # The state is *refined* instead of replaced: x_next = x_t + a*out_proj(attn@v).
        # out_proj is zero-initialised, so at init x_next == x_t exactly and NTE
        # starts as the identity; it can only ever add a refinement.
        self.out_proj = nn.Linear(self.d, self.d)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # learnable gates / scales, all init 1.0 => v13 starts from the previous
        # behaviour and training may shrink or grow each path on its own.
        self.alpha_nte = nn.Parameter(torch.ones(1))   # weight of the attention residual
        self.pos_gain = nn.Parameter(torch.ones(1))    # saccade step size (grid cells)
        self.sim_gain = nn.Parameter(torch.ones(1))    # fidelity dynamic range
        
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
        # identity path = the new frame's own feature at the current gaze centre.
        # With out_proj zero-initialised this makes x_next == that raw feature, so
        # NTE starts as "local attention as a residual on the frame feature" and the
        # per-frame temporal structure is preserved.
        _center = self._get_neighborhood_grid(pos_t, H, W)[:, :, 60, :].unsqueeze(2).to(X_v_next.dtype)
        x_identity = F.grid_sample(X_v_next, _center, mode='bilinear',
                                   align_corners=True).permute(0, 2, 3, 1).squeeze(2)
        x_t_next_evolved = x_identity + self.alpha_nte * self.out_proj((attn @ v).squeeze(2))  # (B, N, D)
        
        # 3. Saccadic Gaze Estimator (位置更新)
        delta_pos = self.pos_gain * torch.tanh(
            self.W_pos(torch.cat([x_t_next_evolved, q.squeeze(2)], dim=-1)))  # (B, N, 2), bounded
        pos_t_next = pos_t + delta_pos
        pos_t_next = torch.stack([pos_t_next[..., 0].clamp(0, W - 1),
                                  pos_t_next[..., 1].clamp(0, H - 1)], dim=-1)
        
        # 4. Adaptive Temporal Decoupling (ATD)
        # bounded similarity: an unbounded W_sim output saturates the sigmoid below
        # (measured: s ~ -5.5 drove c to 0.11, so the ATD threshold fired on EVERY
        # step).  tanh keeps the fidelity in a usable dynamic range instead.
        sim_score = 3.0 * torch.tanh(self.sim_gain * self.W_sim(x_t_next_evolved * x_t))
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


class TracerMambaConnector(STCConnectorV35): 
    """
    整合了 NTE,ATD和BCM的端到端 Tracer。
    不计算 Loss，而是将所需的所有追踪状态作为 loss_data 输出。
    """
    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2, 
                 mv_stages=1, mv_depth=6, window_size=13, num_heads=32, 
                 mlp_ratio=4.0, drop_rate=0.0, attn_drop=0.0, drop_path=0.0, 
                 layer_scale=None, use_attention_every=2):
                 
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        dim = self.encoder_hidden_size  # NTE/BCM 在 vision 维度(1152)做，最后 s1 投影到 LLM
        self.pre_norm = nn.LayerNorm(dim)  # NTE 前归一化（师兄"前面有 layernorm"）
        self.bcm_norm = nn.LayerNorm(dim)  # s1 前归一化：LN(x + BCM(x))
        
        # 1. 挂载 NTE (Neural Trajectory Evolution)
        self.nte = NeuralTrajectoryEvolution(d_model=dim)
        self.nte.apply(self._init_weights)
        # 位置/置信度预测层：W_pos 用小方差让初始位移为 sub-pixel（避免漂移出图，又不落入"不动"平凡解使 loss_cyc 恒 0）；W_sim 零初始化让 fidelity 初始稳定
        nn.init.normal_(self.nte.W_pos.weight, std=1e-3)
        nn.init.zeros_(self.nte.W_pos.bias)
        nn.init.zeros_(self.nte.W_sim.weight)
        nn.init.zeros_(self.nte.W_sim.bias)
        
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

        _last_bcm = layers[-1]

        if isinstance(getattr(_last_bcm, 'downsample_b', None), nn.Identity):
            _last_bcm.downsample_b = nn.Conv2d(_last_bcm.dim2, _last_bcm.dim2, kernel_size=1)
        _seeded = []
        for _name in ('W_fus', 'downsample_b'):
            _mod = getattr(_last_bcm, _name, None)
            if _mod is None or not any(True for _ in _mod.parameters()):
                continue
            for _p in _mod.parameters():
                if _p.dim() > 1:
                    nn.init.normal_(_p, std=1e-3)
                else:
                    nn.init.zeros_(_p)
            _seeded.append(_name)
        if 'W_fus' not in _seeded or 'downsample_b' not in _seeded:
            raise RuntimeError('[BCM] small-init failed on %s: seeded=%s'
                               % (type(_last_bcm).__name__, _seeded))
        print('[BCM] small-std init of %s: %s -> m1(x) is a tiny NON-ZERO residual '
              'at init (pre-norm residual + LayerScale alpha=1)'
              % (type(_last_bcm).__name__, _seeded), flush=True)

        # LayerScale on the BCM residual: s1 now sees LN(x + alpha * BCM(x)).
        # alpha is initialised to 1.0 so training starts exactly as before, then
        # the model can shrink the whole NTE+BCM branch if it is not helpful.
        # It must NOT be initialised to 0: the last BCM layer is zero-initialised,
        # so alpha=0 would zero the gradient into W_fus/downsample_b and the branch
        # would stay dead forever.
        self.alpha = nn.Parameter(torch.ones(1))
        
        
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
        x_res, loss_data, _ = self._temporal_branch(x)
        x_out = self._visual_chain(x_res, self.s1, self.sampler, self.s2, self.readout)
        return x_out, loss_data

    def _temporal_branch(self, x):
        """原 Tracer 时间分支: pre_norm -> NTE(正/反) -> BCM -> 残差。
        返回 (x_res [B,T,D,H,W], loss_data, X_norm [B,T,D,H,W])。
        X_norm 是共享归一化特征 X，Vim 空间分支也读它（方案 §3.1 式2）。"""
        if x.ndim == 4:
            hw = int(x.size(2) ** 0.5)
            x = einops.rearrange(x, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif x.ndim == 5:
            x = einops.rearrange(x, "b t h w d -> b d t h w")

        # NTE 前归一化：稳定 SigLIP 原始特征，避免 W_pos 预测漂移导致 loss_cyc 爆炸
        x = einops.rearrange(x, "b d t h w -> b t h w d")
        x = self.pre_norm(x)
        x = einops.rearrange(x, "b t h w d -> b d t h w")

        B, D, T, H, W = x.shape
        X_v = einops.rearrange(x, "b d t h w -> b t d h w")

        # (1) Neural Trajectory Evolution (前向)
        x_fwd, pos_fwd, c_fwd, i_fwd = self.nte(X_v, reverse=False)

        # (2) Neural Trajectory Evolution (后向循环，用于 Cycle-Consistency)
        pos_init_bwd = pos_fwd[:, :, -1, :].detach()
        _, pos_bwd, _, _ = self.nte(X_v, reverse=True, pos_init=pos_init_bwd)

        # 打包计算 Loss 必需的中间数据
        loss_data = {
            'x_fwd': x_fwd,       # (B, N, T, D) 用于 Semantic Consistency (L_cor)
            'pos_fwd': pos_fwd,   # (B, N, T, 2) 用于 Trajectory Diversity (L_div)
            'c_fwd': c_fwd,       # (B, N, T, 1) Fidelity scores 加权
            'i_fwd': i_fwd,       # (B, N, T, 1) Reset Indicators 截断
            'pos_bwd': pos_bwd,   # (B, N, T, 2) 用于 Cycle-Consistency (L_cyc)
            'alpha': self.alpha.detach(),           # BCM residual scale
            'alpha_nte': self.nte.alpha_nte.detach(),
            'pos_gain': self.nte.pos_gain.detach(),
            'sim_gain': self.nte.sim_gain.detach(),
        }

        # (3) BCM 建模 (MambaVision)
        # BiMambaVisionLayer 期望 5D 输入 (b, t, d, h, w)
        # 注意：此处 x_fwd 的 N 维度本质上是展平的 H*W
        x_mamba_in = einops.rearrange(x_fwd, "b (h w) t d -> b t d h w", h=H, w=W)
        x_mamba_out = torch.utils.checkpoint.checkpoint(self.m1, x_mamba_in, use_reentrant=False)

        # (3.5) 残差 + LayerNorm：s1 吃 LN(x + BCM(x))，让 s1 输入分布稳定
        # pre-norm residual: normalise the BASE only, then add the gated branch.
        # Normalising the sum would let the LayerNorm Jacobian cancel the small
        # branch signal (measured: W_fus stayed at 1.4e-4 over a whole epoch).
        x_base = einops.rearrange(x_mamba_in, "b t d h w -> b t h w d")
        x_base = self.bcm_norm(x_base)
        x_base = einops.rearrange(x_base, "b t h w d -> b t d h w")
        x_res = x_base + self.alpha * x_mamba_out

        return x_res, loss_data, X_v

    def _visual_chain(self, z, s1, sampler, s2, readout):
        """视觉链 P_phi: s1 -> sampler -> s2 -> readout。
        z: [B, T, D, H, W] -> [B, L, 3584]；时间/空间两路结构相同、参数独立(各自传入)。"""
        t = z.size(1)
        x_proj = einops.rearrange(z, "b t d h w -> (b t) d h w")
        x_proj = s1(x_proj)                      # (b*t, 3584, H, W)
        # temporal + spatial downsampling, then readout
        x_proj = einops.rearrange(x_proj, "(b t) d h w -> b d t h w", t=t)
        x_proj = sampler(x_proj)                 # Conv3d k=s=(2,2,2), padding=0
        t_ds = x_proj.size(2)
        x_proj = einops.rearrange(x_proj, "b d t h w -> (b t) d h w")
        x_proj = s2(x_proj)                      # RegStage(3584 -> 3584)
        x_out = einops.rearrange(x_proj, "(b t) d h w -> b (t h w) d", t=t_ds)
        x_out = readout(x_out)
        return x_out


class RMSNorm(nn.Module):
    """方案 §4.1/式6 的 RMSNorm：不跨 patch/帧、不减均值、可学习缩放初始 1、eps=1e-5。"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, s):
        # s: (..., dim)，沿通道维归一化
        rms = (s.float().pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        return (s / rms.to(s.dtype)) * self.g


class CrossAttentionFusion(nn.Module):
    """方案 v1 §4：时间 token 作 Query、空间 token 作 Key/Value 的单层四头交叉注意力。
    输出长度 = Query 数 = 1352（固定 LLM 视觉 token 预算，不随两路拼接变 2704）。
    F = T + gamma_fuse * U，U = Concat_{heads}(O) W_O。
    """
    def __init__(self, dim=3584, n_heads=4, head_dim=64, eps=1e-5, gamma_fuse=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner = n_heads * head_dim   # 256
        # gamma_fuse 是普通 float（非 nn.Parameter），避免新增标量参数触发 alpha_sp 式 cuDNN 崩溃。
        self.gamma_fuse = gamma_fuse
        self.ln_q = nn.LayerNorm(dim, eps=eps)     # 缩放初始 1、偏置 0（默认）
        self.ln_kv = nn.LayerNorm(dim, eps=eps)
        self.w_q = nn.Linear(dim, self.inner, bias=False)
        self.w_k = nn.Linear(dim, self.inner, bias=False)
        self.w_v = nn.Linear(dim, self.inner, bias=False)
        self.w_o = nn.Linear(self.inner, dim, bias=False)
        # §5.2：四个融合投影 Xavier uniform 非零初始化（不同时置零残差系数）。
        for _m in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.xavier_uniform_(_m.weight)

    def forward(self, T, S):
        # T, S: [B, N, D]
        B, N, D = T.shape
        Tq = self.ln_q(T)
        Sk = self.ln_kv(S)
        q = self.w_q(Tq).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)   # [B,H,N,dh]
        k = self.w_k(Sk).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(Sk).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)       # [B,H,N,N]
        attn = F.softmax(attn, dim=-1)                                             # softmax 沿空间位置 j
        o = torch.matmul(attn, v)                                                  # [B,H,N,dh]
        o = o.transpose(1, 2).contiguous().view(B, N, self.inner)                  # [B,N,256]
        U = self.w_o(o)                                                            # [B,N,D]
        return T + self.gamma_fuse * U


class VimSpatialConnector(TracerMambaConnector):
    """Tracer + Vim 单块空间分支（方案 2026-10-31）。

    时间分支(NTE+BCM+P_T)完全不变；新增空间分支：
      - 每帧 27×27 按行优先展开成 729 序列（帧间独立，方案 §3.2 式3-4）；
      - 过一个 Vim BiMamba v2 单块（RMSNorm + 单次残差，方案 §4.2 式7-9）；
      - 恢复 27×27 网格后走独立空间适配器 P_S（复制 s1/sampler/s2/readout 初值，参数独立，§6.2）；
      - 时间/空间各 1352 token，沿序列维拼接成 2704（方案 §7.1 式14）。
    """
    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2,
                 mv_stages=1, mv_depth=6, window_size=13, num_heads=32,
                 mlp_ratio=4.0, drop_rate=0.0, attn_drop=0.0, drop_path=0.0,
                 layer_scale=None, use_attention_every=2, spatial_only=False, fusion=False,
                 joint=False):
        super().__init__(config=config, downsample=downsample, depth=depth,
                         mlp_depth=mlp_depth, mv_stages=mv_stages, mv_depth=mv_depth,
                         window_size=window_size, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop=attn_drop,
                         drop_path=drop_path, layer_scale=layer_scale,
                         use_attention_every=use_attention_every)
        dim = self.encoder_hidden_size  # 1152
        # Stage1 空间单分支：只跑 pre_norm -> sp_norm -> Vim -> P_S，返回 1352 token、无 loss_data。
        self.spatial_only = spatial_only
        # 阶段 B：时间查空间交叉注意力融合（固定 1352 token），否则 2704 拼接。
        self.fusion = fusion
        # 联合训练：两路分支 + 融合 + LoRA 同阶段训练，forward 不包 no_grad、传回 loss_data。
        self.joint = joint

        # 单块 Vim BiMamba v2（方案 §4.1：d_state=16 d_conv=4 expand=2 if_divide_out=True）
        from videollama2.model.vim_mamba_simple import Mamba
        self.sp_norm = RMSNorm(dim, eps=1e-5)
        self.sp_mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2,
                              dt_rank='auto', bias=False, conv_bias=True,
                              bimamba_type='v2', if_devide_out=True)

        # 残差缩放起步（2026-11-05 修正）：out_proj 小尺度非零初始化，打破零初始化的
        # 「内部参数梯度恒为 0」锁死（梯度检查坐实 dΔ/d内部 = out_proj.weight = 0）。
        # normal(1e-3) 让内部参数立刻有非零梯度；lambda_sp 是普通 float（非 Parameter，
        # 不改变 named_parameters 顺序、不触发 alpha_sp 式 cuDNN 崩溃）压住初始 Δ 幅度防退化。
        nn.init.normal_(self.sp_mamba.out_proj.weight, std=1e-3)
        if self.sp_mamba.out_proj.bias is not None:
            nn.init.zeros_(self.sp_mamba.out_proj.bias)
        self.lambda_sp = 0.1

        # 阶段 B 融合模块：时间查空间，内部 256、四头各 64，输出 3584（与两路 token 同宽）。
        if fusion:
            self.fuse = CrossAttentionFusion(dim=self.hidden_size)

        # 独立空间适配器 P_S：复制时间侧 s1/sampler/s2/readout 初值（方案 §6.2 步骤2）
        self.s1_s = copy.deepcopy(self.s1)
        self.sampler_s = copy.deepcopy(self.sampler)
        self.s2_s = copy.deepcopy(self.s2)
        self.readout_s = copy.deepcopy(self.readout)

        # 加载顺序(方案 §6.2 步骤2/5)：加载预训练 mm_projector.bin(无空间参数)后，
        # 把时间侧初值复制到空间侧 P_S；加载 checkpoint(含空间参数)时直接加载、不再复制。
        self.register_load_state_dict_pre_hook(self._detect_spatial_in_sd)
        self.register_load_state_dict_post_hook(self._copy_spatial_after_load)

    def _detect_spatial_in_sd(self, module, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._spatial_in_sd = any(
            k.startswith(prefix + 's1_s.') or k.startswith(prefix + 'sp_mamba.')
            or k.startswith(prefix + 'sp_norm.') or k.startswith(prefix + 'sampler_s.')
            for k in state_dict
        )

    def _copy_spatial_after_load(self, module, incompatible_keys):
        if getattr(self, '_spatial_in_sd', False):
            return
        self.s1_s.load_state_dict(self.s1.state_dict())
        self.sampler_s.load_state_dict(self.sampler.state_dict())
        self.s2_s.load_state_dict(self.s2.state_dict())
        self.readout_s.load_state_dict(self.readout.state_dict())

    def _spatial_branch(self, X_norm):
        """X_norm [B,T,D,H,W] -> 每帧行优先展开 729 -> Vim 双向块 -> 恢复网格 -> FS [B,T,D,H,W]。
        方案 §3.2 式3-4、§3.3 式5、§4.2 式7-9。"""
        B, T, D, H, W = X_norm.shape
        # 式3：n = H*h + w 行优先；合并 batch 维成 (B*T, 729, D)
        S = X_norm.permute(0, 1, 3, 4, 2).reshape(B * T, H * W, D)
        # 式7-9：U = RMSNorm(S); Δ = BiMamba(U); G = S + λ·Δ（λ 压住初始扰动防退化）
        U = self.sp_norm(S)
        delta = self.sp_mamba(U)
        G = S + self.lambda_sp * delta
        # 式5：恢复 [B,T,H,W,D] -> [B,T,D,H,W]
        FS = G.reshape(B, T, H, W, D).permute(0, 1, 4, 2, 3)
        return FS

    def _pre_norm_features(self, x):
        """pre_norm only -> X_v（共享归一化特征）。Stage1 空间单分支只用这一路，
        不经过 NTE/BCM/时间视觉链；与 _temporal_branch 开头完全一致。"""
        if x.ndim == 4:
            hw = int(x.size(2) ** 0.5)
            x = einops.rearrange(x, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif x.ndim == 5:
            x = einops.rearrange(x, "b t h w d -> b d t h w")
        x = einops.rearrange(x, "b d t h w -> b t h w d")
        x = self.pre_norm(x)
        x = einops.rearrange(x, "b t h w d -> b d t h w")
        return einops.rearrange(x, "b d t h w -> b t d h w")

    def forward(self, x):
        if self.spatial_only:
            # Stage1：只跑空间分支，时间分支(NTE/BCM/P_T)不参与前向，无 NTE 辅助损失。
            X_norm = self._pre_norm_features(x)
            FS = self._spatial_branch(X_norm)
            VS = self._visual_chain(FS, self.s1_s, self.sampler_s, self.s2_s, self.readout_s)
            return VS, None
        if self.fusion:
            if self.joint:
                # 联合训练：时间/空间两路都保留计算图；返回 temporal 的 loss_data
                # 供 NTE 辅助损失（L_cor/L_cyc/L_div）继续生效。
                x_res, loss_data, X_norm = self._temporal_branch(x)
                VT = self._visual_chain(x_res, self.s1, self.sampler, self.s2, self.readout)
                FS = self._spatial_branch(X_norm)
                VS = self._visual_chain(FS, self.s1_s, self.sampler_s, self.s2_s, self.readout_s)
                F = self.fuse(VT, VS)   # [B, 1352, 3584]
                return F, loss_data
            # 阶段 B：两路分支冻结，no_grad 生成 T/S（省梯度显存，方案 §5.1）；
            # 融合计算与 LLM 前向保留计算图。无 NTE 辅助损失。
            with torch.no_grad():
                x_res, _loss_data, X_norm = self._temporal_branch(x)
                VT = self._visual_chain(x_res, self.s1, self.sampler, self.s2, self.readout)
                FS = self._spatial_branch(X_norm)
                VS = self._visual_chain(FS, self.s1_s, self.sampler_s, self.s2_s, self.readout_s)
            F = self.fuse(VT, VS)   # [B, 1352, 3584]
            return F, None
        x_res, loss_data, X_norm = self._temporal_branch(x)
        VT = self._visual_chain(x_res, self.s1, self.sampler, self.s2, self.readout)
        FS = self._spatial_branch(X_norm)
        VS = self._visual_chain(FS, self.s1_s, self.sampler_s, self.s2_s, self.readout_s)
        # 式14：先时间后空间沿 token 维拼接 -> [B, 2704, 3584]
        return torch.cat([VT, VS], dim=1), loss_data


# =============================================================================
#  Tracer + Vim Cross-Mamba + RFM（方案 v2，2026-11-11）
#
#  §2 坐标对齐：NTE 实际采样中心 -> 在空间特征 S 上双线性取样
#  §3 逐帧 Cross-Mamba：GuidedScan(main, guide) —— B/C/Δ 来自引导路，A/D 是参数
#  §4/§5 RFM：两路各自 Q/K/V + 共享 softmax 的关系融合，旁路卷积接回 B0 接口
#  方案字段（由 config 传入）：st_fusion_mode / st_scan_mode / st_align_mode /
#  st_fusion_scale / st_rank_weight / st_frame_chunk
# =============================================================================


def align_spatial_features(S5, pos_fwd, i_fwd, mode='nte_center'):
    """§2.2 式(2)：把空间特征对齐到时间轨迹槽位。

    S5      : [B, T, D, H, W]   逐帧 Vim 输出（真实 patch 网格）
    pos_fwd : [B, N, T, 2]      NTE 每帧更新后的坐标 (x, y)，N=H*W，槽位行优先
    i_fwd   : [B, N, T, 1]      该帧 ATD 重置指示
    返回    : [B*T, N, D]

    普通跟踪用上一帧保存的坐标 p_{t-1}；ATD 重置用本帧更新后的坐标 p_t；
    首帧用初始坐标（pos_fwd[:, :, 0]）。
    mode='slot' 时完全不采样，直接按初始槽位行优先展开（A5 对照）。
    """
    B, T, D, H, W = S5.shape
    N = H * W
    if mode == 'slot':
        return S5.permute(0, 1, 3, 4, 2).reshape(B * T, N, D)

    assert mode == 'nte_center', f'unknown st_align_mode: {mode}'
    pos = pos_fwd.permute(0, 2, 1, 3).contiguous()          # [B, T, N, 2]
    rst = i_fwd.permute(0, 2, 1, 3).to(torch.bool)          # [B, T, N, 1]
    if T > 1:
        prev = torch.cat([pos[:, :1], pos[:, :-1]], dim=1)  # p_{t-1}, 首帧用自身
    else:
        prev = pos
    center = torch.where(rst, pos, prev)                    # [B, T, N, 2]

    gx = 2.0 * center[..., 0] / max(W - 1, 1) - 1.0
    gy = 2.0 * center[..., 1] / max(H - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).reshape(B * T, N, 1, 2)

    # S5 已是 [B,T,D,H,W]；合并帧维后直接作为 grid_sample 的 input
    maps = S5.reshape(B * T, D, H, W)
    samp = F.grid_sample(maps.float(), grid.float(), mode='bilinear',
                         padding_mode='border', align_corners=True)   # [B*T, D, N, 1]
    return samp.squeeze(-1).permute(0, 2, 1).to(S5.dtype)    # [B*T, N, D]


class GuidedScan(nn.Module):
    """§3/§11.1 引导式选择性扫描（Cross-Mamba 单向）。

    main, guide : [N, L, D]（N = B*F 帧，L = 729）
    返回 delta  : [N, L, D]

    内部 d_m=256、状态 n=16：
      u   = Conv1d_leftpad3( in_main(LN(main)) )
      g   = in_guide(LN(guide))
      B,C = b_proj(g), c_proj(g);  Δ = softplus(dt_proj(g) + b_Δ)
      h_{i,c,k} = exp(Δ_{i,c} A_{c,k}) h_{i-1,c,k} + Δ_{i,c} B_{i,k} u_{i,c}
      y_{i,c}   = Σ_k C_{i,k} h_{i,c,k} + D_c u_{i,c}
      delta = out_proj( LN_out(y) * SiLU(gate_proj(main)) )
    A/D 为可学习参数（M³amba Algorithm 1），Δ 的 softplus 只做一次。
    """
    def __init__(self, d_model=1152, d_m=256, d_state=16, d_conv=3,
                 dt_min=1e-3, dt_max=1e-1, out_std=1e-3):
        super().__init__()
        self.d_model = d_model
        self.d_m = d_m
        self.d_state = d_state
        self.d_conv = d_conv
        self.ln_main = nn.LayerNorm(d_model, eps=1e-5)
        self.ln_guide = nn.LayerNorm(d_model, eps=1e-5)
        self.in_main = nn.Linear(d_model, d_m, bias=False)
        self.in_guide = nn.Linear(d_model, d_m, bias=False)
        self.gate_proj = nn.Linear(d_model, d_m, bias=True)
        self.b_proj = nn.Linear(d_m, d_state, bias=False)
        self.c_proj = nn.Linear(d_m, d_state, bias=False)
        self.dt_proj = nn.Linear(d_m, d_m, bias=False)     # 无 bias，bias 单独做 delta_bias
        self.dt_bias = nn.Parameter(torch.zeros(d_m))
        # A_log 每通道复制 log(1..n) -> A = -exp(A_log) = -(1..n)
        a_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        self.A_log = nn.Parameter(a_log.repeat(d_m, 1).contiguous())
        self.D = nn.Parameter(torch.ones(d_m))
        self.conv1d = nn.Conv1d(d_m, d_m, kernel_size=d_conv, groups=d_m,
                                padding=0, bias=True)
        self.ln_out = nn.LayerNorm(d_m, eps=1e-5)
        self.out_proj = nn.Linear(d_m, d_model, bias=True)
        self._reset_parameters(dt_min, dt_max, out_std)

    def _reset_parameters(self, dt_min, dt_max, out_std):
        # Δ 的校准时区：softplus^{-1}(U[dt_min, dt_max])
        dt = torch.exp(torch.rand(self.d_m) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=1e-4)
        inv = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_bias.copy_(inv)
            self.D.fill_(1.0)
        nn.init.normal_(self.out_proj.weight, std=out_std)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, main, guide):
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        N, L, D = main.shape
        m = self.ln_main(main)
        g = self.ln_guide(guide)

        u = self.in_main(m).transpose(1, 2)                 # [N, d_m, L]
        u = F.pad(u, (self.d_conv - 1, 0))                  # 左侧补零（因果）
        u = self.conv1d(u)[..., :L].contiguous()

        gate = self.gate_proj(m)                            # [N, L, d_m]
        gg = self.in_guide(g)                               # [N, L, d_m]
        Bv = self.b_proj(gg).transpose(1, 2).contiguous()   # [N, n, L]
        Cv = self.c_proj(gg).transpose(1, 2).contiguous()   # [N, n, L]
        dt = self.dt_proj(gg).transpose(1, 2).contiguous()  # [N, d_m, L]

        A = -torch.exp(self.A_log.float())                  # [d_m, n]
        y = selective_scan_fn(u, dt, A, Bv, Cv, self.D.float(),
                              z=None, delta_bias=self.dt_bias.float(),
                              delta_softplus=True)
        y = y.transpose(1, 2)                               # [N, L, d_m]
        delta = self.out_proj(self.ln_out(y) * F.silu(gate))
        return delta


class PointwiseFusion(nn.Module):
    """A1：逐位置 2304 -> 1152 线性层（无 top conv / FFN）。
    初值 [I, 0]、偏置 0 → 初始 Z = XT（时间主路），配合 γ 接回 B0。"""
    def __init__(self, dim=1152):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim, bias=True)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[:, :dim].copy_(torch.eye(dim))
            self.proj.bias.zero_()

    def forward(self, XT, XS):
        return self.proj(torch.cat([XT, XS], dim=-1))


class RFMFusion(nn.Module):
    """§4/§12.1 关系融合模块（Reinforce Fusion Module）。

    XT, XS : [N, L, D]
    返回   : Z [N, L, D], rank_penalty（st_rank_weight>0 时，否则 None）

    自读用本路权重 P_m，交叉读用共享 P=softmax((E_T+E_S)/2)；
    LN 位置按 M³amba 式(4)：LN(self 读出) + cross 读出。
    """
    def __init__(self, dim=1152, n_heads=4, head_dim=64, eps=1e-5,
                 frame_chunk=2, use_ckpt=False, init_std=1e-3):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner = n_heads * head_dim          # 256
        self.frame_chunk = frame_chunk
        self.use_ckpt = use_ckpt
        self.ln_t = nn.LayerNorm(dim, eps=eps)
        self.ln_s = nn.LayerNorm(dim, eps=eps)
        self.qkv_t = nn.Linear(dim, 3 * self.inner, bias=True)
        self.qkv_s = nn.Linear(dim, 3 * self.inner, bias=True)
        self.ln_self_t = nn.LayerNorm(self.inner, eps=eps)
        self.ln_self_s = nn.LayerNorm(self.inner, eps=eps)
        self.pointwise_top = nn.Linear(2 * self.inner, dim, bias=True)   # 512 -> 1152
        self.pointwise_skip = nn.Linear(2 * dim, dim, bias=True)         # 2304 -> 1152
        self.ln_fuse = nn.LayerNorm(dim, eps=eps)
        self.ffn1 = nn.Linear(dim, 2 * dim, bias=True)                   # 1152 -> 2304
        self.ffn2 = nn.Linear(2 * dim, dim, bias=True)                   # 2304 -> 1152
        self._reset_parameters(init_std)

    def _reset_parameters(self, init_std):
        # 旁路卷积 [I, 0]、偏置 0 → 初始 Z = XT
        with torch.no_grad():
            self.pointwise_skip.weight.zero_()
            self.pointwise_skip.weight[:, :self.dim].copy_(torch.eye(self.dim))
            self.pointwise_skip.bias.zero_()
        # 上方卷积 / FFN 末层小偏非零权重
        nn.init.normal_(self.pointwise_top.weight, std=init_std)
        nn.init.zeros_(self.pointwise_top.bias)
        nn.init.normal_(self.ffn2.weight, std=init_std)
        nn.init.zeros_(self.ffn2.bias)

    def _split_heads(self, x):
        # [N, L, 3*inner] -> q,k,v [N, H, L, dh]
        N, L, _ = x.shape
        q, k, v = x.split(self.inner, dim=-1)
        q = q.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _merge_heads(self, x):
        # [N, H, L, dh] -> [N, L, H*dh]
        N, H, L, dh = x.shape
        return x.transpose(1, 2).contiguous().view(N, L, H * dh)

    def _chunk(self, XT, XS, need_rank=False):
        qt, kt, vt = self._split_heads(self.qkv_t(self.ln_t(XT)))
        qs, ks, vs = self._split_heads(self.qkv_s(self.ln_s(XS)))
        scale = self.head_dim ** -0.5
        et = torch.matmul(qt.float(), kt.float().transpose(-1, -2)) * scale
        es = torch.matmul(qs.float(), ks.float().transpose(-1, -2)) * scale
        pt = torch.softmax(et, dim=-1)
        ps = torch.softmax(es, dim=-1)
        p = torch.softmax((et + es) * 0.5, dim=-1)          # 共享融合权重

        # 自读（本路 P_m，LN 包住自读输出） + 交叉读（共享 P）
        self_t = self._merge_heads(torch.matmul(pt.to(vt.dtype), vt))
        self_s = self._merge_heads(torch.matmul(ps.to(vs.dtype), vs))
        cross_t = self._merge_heads(torch.matmul(p.to(vs.dtype), vs))
        cross_s = self._merge_heads(torch.matmul(p.to(vt.dtype), vt))
        tt = self.ln_self_t(self_t) + cross_t
        ss = self.ln_self_s(self_s) + cross_s

        top = self.pointwise_top(torch.cat([tt, ss], dim=-1))
        skip = self.pointwise_skip(torch.cat([XT, XS], dim=-1))
        u = top + skip
        z = u + self.ffn2(F.gelu(self.ffn1(self.ln_fuse(u))))

        rank = None
        if need_rank:
            # M³amba 式(7)：对 E 的 frontal slices 求核范数（跨头/两路求和）
            rank = (torch.linalg.matrix_norm(et, ord='nuc').sum()
                    + torch.linalg.matrix_norm(es, ord='nuc').sum())
        return z, rank

    def forward(self, XT, XS, b=1):
        N, L, _ = XT.shape
        chunk = max(1, int(self.frame_chunk) * max(1, int(b)))
        need_rank = getattr(self, 'rank_weight', 0.0) > 0.0
        zs, ranks = [], []
        for s in range(0, N, chunk):
            xt, xs = XT[s:s + chunk], XS[s:s + chunk]
            if self.use_ckpt and self.training:
                z, r = torch.utils.checkpoint.checkpoint(self._chunk, xt, xs,
                                                         use_reentrant=False)
            else:
                z, r = self._chunk(xt, xs, need_rank=need_rank)
            zs.append(z)
            if r is not None:
                ranks.append(r)
        Z = torch.cat(zs, dim=0)
        rank = None
        if ranks:
            rank = torch.stack(ranks).sum()
            n_frames = N // max(1, int(b))
            # §5.3：分母 B * F * 4 * 2 * L
            rank = rank / float(max(1, b) * max(1, n_frames) * self.n_heads * 2 * L)
        return Z, rank


class SpatioTemporalCrossConnector(TracerMambaConnector):
    """Tracer 时间分支 + 逐帧 Vim 空间分支 + Cross-Mamba + RFM（方案 v2）。

    st_fusion_mode : 'off'(B0) | 'conv'(A1) | 'rfm'(A2-A5)
    st_scan_mode   : 'off' | 'self'(A3) | 'cross'(A4/A5)
    st_align_mode  : 'nte_center'(A1-A4) | 'slot'(A5)
    st_fusion_scale: γ（默认 0.1，固定 float）
    st_rank_weight : 低秩正则权重（A1-A5 为 0）
    st_frame_chunk : RFM 分块大小（按 b 个视频的帧数）
    """
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        dim = self.encoder_hidden_size

        self.st_fusion_mode = str(getattr(config, 'st_fusion_mode', 'off'))
        self.st_scan_mode = str(getattr(config, 'st_scan_mode', 'off'))
        self.st_align_mode = str(getattr(config, 'st_align_mode', 'nte_center'))
        self.st_fusion_scale = float(getattr(config, 'st_fusion_scale', 0.1))
        self.st_rank_weight = float(getattr(config, 'st_rank_weight', 0.0))
        self.st_frame_chunk = int(getattr(config, 'st_frame_chunk', 2))
        if self.st_fusion_mode not in ('off', 'conv', 'rfm'):
            raise ValueError(f'unknown st_fusion_mode: {self.st_fusion_mode}')
        if self.st_scan_mode not in ('off', 'self', 'cross'):
            raise ValueError(f'unknown st_scan_mode: {self.st_scan_mode}')
        if self.st_align_mode not in ('nte_center', 'slot'):
            raise ValueError(f'unknown st_align_mode: {self.st_align_mode}')
        if self.st_scan_mode == 'cross' and self.st_align_mode == 'slot':
            pass  # A5 合法：cross + slot

        # ---- 逐帧 Vim 空间分支（§2.3，与 VimSpatialConnector 同一实现）----
        from videollama2.model.vim_mamba_simple import Mamba
        self.sp_norm = RMSNorm(dim, eps=1e-5)
        self.sp_mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2,
                              dt_rank='auto', bias=False, conv_bias=True,
                              bimamba_type='v2', if_devide_out=True)
        nn.init.normal_(self.sp_mamba.out_proj.weight, std=1e-3)
        if self.sp_mamba.out_proj.bias is not None:
            nn.init.zeros_(self.sp_mamba.out_proj.bias)
        self.lambda_sp = 0.1

        # ---- 逐帧 Cross-Mamba（§3）：两次扫描，同一行优先顺序，角色互换 ----
        if self.st_scan_mode != 'off':
            self.scan_ts = GuidedScan(dim)   # main=T, guide=S
            self.scan_st = GuidedScan(dim)   # main=S, guide=T

        # ---- 融合（§4/§5）----
        if self.st_fusion_mode == 'conv':
            self.fuse = PointwiseFusion(dim)
        elif self.st_fusion_mode == 'rfm':
            ckpt = os.environ.get('ST_RFM_CKPT', '0') == '1'
            self.fuse = RFMFusion(dim, frame_chunk=self.st_frame_chunk,
                                  use_ckpt=ckpt)
            self.fuse.rank_weight = self.st_rank_weight

        if os.environ.get('LOCAL_RANK', '0') in ('0', '-1'):
            print('[ST] fusion_mode=%s scan_mode=%s align_mode=%s gamma=%.3f '
                  'rank=%.1e frame_chunk=%d'
                  % (self.st_fusion_mode, self.st_scan_mode, self.st_align_mode,
                     self.st_fusion_scale, self.st_rank_weight, self.st_frame_chunk),
                  flush=True)

    def _spatial_branch(self, X_norm):
        """与 VimSpatialConnector._spatial_branch 完全一致（§2.3/§4.2）。"""
        B, T, D, H, W = X_norm.shape
        S = X_norm.permute(0, 1, 3, 4, 2).reshape(B * T, H * W, D)
        U = self.sp_norm(S)
        delta = self.sp_mamba(U)
        G = S + self.lambda_sp * delta
        return G.reshape(B, T, H, W, D).permute(0, 1, 4, 2, 3)

    def forward(self, x):
        # 关闭新路径（或 γ=0）：与原 B0（TracerMambaConnector）逐位一致
        if self.st_fusion_mode == 'off' or self.st_fusion_scale == 0.0:
            return super().forward(x)

        x_res, loss_data, X_norm = self._temporal_branch(x)
        B, T, D, H, W = x_res.shape
        N = H * W

        # §2：首步打印真实样本的旧坐标/新坐标/ATD 标记/实际采样中心（用户要求）
        if (os.environ.get('ST_DEBUG_COORD', '0') == '1'
                and os.environ.get('LOCAL_RANK', '0') in ('0', '-1')
                and not getattr(self, '_coord_dbg_printed', False)):
            self._coord_dbg_printed = True
            with torch.no_grad():
                _pos = loss_data['pos_fwd'].permute(0, 2, 1, 3)   # [B,T,N,2]
                _rst = loss_data['i_fwd'].permute(0, 2, 1, 3)
                _prev = torch.cat([_pos[:, :1], _pos[:, :-1]], dim=1)
                _center = torch.where(_rst.bool(), _pos, _prev)
                for _t in [0, 1, min(2, T - 1)]:
                    for _s in [0, 1, N // 2]:
                        print('[ST-COORD] b=0 t=%d slot=%d old=(%.3f,%.3f) new=(%.3f,%.3f) '
                              'reset=%d center=(%.3f,%.3f)'
                              % (_t, _s,
                                 _prev[0, _t, _s, 0].item(), _prev[0, _t, _s, 1].item(),
                                 _pos[0, _t, _s, 0].item(), _pos[0, _t, _s, 1].item(),
                                 int(_rst[0, _t, _s, 0].item()),
                                 _center[0, _t, _s, 0].item(), _center[0, _t, _s, 1].item()),
                              flush=True)

        # §2：空间编码 + 坐标对齐
        S5 = self._spatial_branch(X_norm)                       # [B,T,D,H,W]
        S_al = align_spatial_features(S5, loss_data['pos_fwd'],
                                      loss_data['i_fwd'], self.st_align_mode)
        T_flat = x_res.permute(0, 1, 3, 4, 2).reshape(B * T, N, D)
        XT, XS = T_flat, S_al

        # §3：逐帧双向 Cross-Mamba（角色互换，两次同一行优先顺序）
        if self.st_scan_mode == 'self':
            XT = XT + self.scan_ts(XT, XT)
            XS = XS + self.scan_st(XS, XS)
        elif self.st_scan_mode == 'cross':
            dT = self.scan_ts(XT, XS)
            dS = self.scan_st(XS, XT)
            XT = XT + dT
            XS = XS + dS

        # §4/§5：融合 + γ 接回 B0（残差用扫描前的 T）
        rank = None
        if self.st_fusion_mode == 'conv':
            Z = self.fuse(XT, XS)
        else:
            Z, rank = self.fuse(XT, XS, b=B)
        F_flat = T_flat + self.st_fusion_scale * (Z - T_flat)
        Fg = F_flat.reshape(B, T, H, W, D).permute(0, 1, 4, 2, 3)

        out = self._visual_chain(Fg, self.s1, self.sampler, self.s2, self.readout)

        if rank is not None and self.st_rank_weight > 0:
            loss_data = dict(loss_data)
            loss_data['st_rank_loss'] = rank
        if os.environ.get('ST_DEBUG', '0') == '1':
            loss_data = dict(loss_data)
            loss_data['st_delta_ratio'] = (
                (self.st_fusion_scale * (Z - T_flat)).float().norm()
                / T_flat.float().norm().clamp(min=1e-6))
        return out, loss_data