#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""方案 v2 自检：GuidedScan / RFM / A1 融合 / 坐标对齐。

用法（GPU 2/3 空闲时）：
    CUDA_VISIBLE_DEVICES=3 python -u tools/verify_cross_rfm.py

检查项（对齐方案 §10.2 / §14.1）：
  1. GuidedScan 与逐时间步参考递推一致（验证 B/C/Δ 来自引导路、A/D 参数化）
  2. RFM 形状 / skip=[I,0] 初始近恒等 / 梯度有限非零 / 低秩项可微
  3. PointwiseFusion 初始化严格等于 XT
  4. 坐标对齐：slot 行优先、静态坐标等于原特征、移动/ATD 重置取对坐标
  5. 参数量统计
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videollama2.model.projector import (  # noqa: E402
    GuidedScan, RFMFusion, PointwiseFusion, align_spatial_features,
)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
FAIL = []


def check(name, ok, detail=''):
    print(('  PASS  ' if ok else '  FAIL  ') + name + (('  | ' + detail) if detail else ''))
    if not ok:
        FAIL.append(name)


def reference_scan(module, main, guide):
    """逐时间步参考实现（float32），与 GuidedScan.forward 的算子顺序一致。"""
    m = module.ln_main(main.float())
    g = module.ln_guide(guide.float())
    u = module.in_main(m).transpose(1, 2)                       # [N,dm,L]
    u = F.pad(u, (module.d_conv - 1, 0))
    u = module.conv1d(u.float())[..., :main.shape[1]]           # [N,dm,L]
    gate = module.gate_proj(m)
    gg = module.in_guide(g)
    Bv = module.b_proj(gg)                                      # [N,L,n]
    Cv = module.c_proj(gg)
    dt = module.dt_proj(gg) + module.dt_bias                    # [N,L,dm]
    dt = F.softplus(dt.float())
    A = -torch.exp(module.A_log.float())                        # [dm,n]
    D = module.D.float()
    N, L, dm = u.shape[0], u.shape[2], u.shape[1]
    h = torch.zeros(N, dm, module.d_state, device=u.device)
    ys = []
    for t in range(L):
        d_t = dt[:, t, :]                                       # [N,dm]
        h = torch.exp(d_t.unsqueeze(-1) * A.unsqueeze(0)) * h \
            + d_t.unsqueeze(-1) * u[:, :, t].unsqueeze(-1) * Bv[:, t].unsqueeze(1)
        y_t = (h * Cv[:, t].unsqueeze(1)).sum(-1) + D.unsqueeze(0) * u[:, :, t]
        ys.append(y_t)
    y = torch.stack(ys, 1)                                      # [N,L,dm]
    delta = module.out_proj(module.ln_out(y) * F.silu(gate))
    return delta


def test_guided_scan():
    print('\n[1] GuidedScan vs 参考递推')
    torch.manual_seed(0)
    d_model, d_m, d_state = 64, 16, 4
    mod = GuidedScan(d_model=d_model, d_m=d_m, d_state=d_state).to(DEV).float()
    main = torch.randn(3, 11, d_model, device=DEV)
    guide = torch.randn(3, 11, d_model, device=DEV)
    with torch.no_grad():
        out = mod(main, guide)
        ref = reference_scan(mod, main, guide)
    err = (out - ref).abs().max().item()
    check('scan 输出与参考递推一致', err < 1e-5, 'max|diff|=%.2e' % err)

    # 引导路只影响 B/C/Δ：交换 guide 必须改变输出
    with torch.no_grad():
        out2 = mod(main, guide.flip(1))
    dg = (out - out2).abs().max().item()
    check('guide 改变会改变输出', dg > 1e-6, 'max|diff|=%.2e' % dg)

    # 梯度有限且 A/D 有梯度
    mod.zero_grad()
    main.requires_grad_(True)
    mod(main, guide).sum().backward()
    gA = mod.A_log.grad
    gD = mod.D.grad
    check('A/D 有非零梯度', gA is not None and gD is not None
          and torch.isfinite(gA).all() and gA.abs().sum() > 0 and gD.abs().sum() > 0)


def test_rfm():
    print('\n[2] RFM')
    torch.manual_seed(0)
    dim, N, L = 1152, 4, 29
    mod = RFMFusion(dim).to(DEV).float()
    XT = torch.randn(N, L, dim, device=DEV)
    XS = torch.randn(N, L, dim, device=DEV)
    Z, rank = mod(XT, XS, b=2)
    check('输出形状', tuple(Z.shape) == (N, L, dim), str(tuple(Z.shape)))
    rel = ((Z - XT).norm() / XT.norm()).item()
    check('初始 Z≈XT（skip=[I,0]）', rel < 0.10, 'rel=%.4f' % rel)

    mod.zero_grad()
    Z.sum().backward()
    bad = [n for n, p in mod.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    nz = [n for n, p in mod.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    check('所有梯度有限', len(bad) == 0, bad[:3].__str__())
    check('所有参数梯度非零', len(nz) == len(list(mod.named_parameters())),
          '%d/%d' % (len(nz), len(list(mod.named_parameters()))))
    with torch.no_grad():
        w = mod.pointwise_skip.weight
        check('skip 时间半边=单位阵', torch.allclose(w[:, :dim], torch.eye(dim, device=DEV), atol=1e-6))
        check('skip 空间半边=0', w[:, dim:].abs().max().item() == 0.0)

    mod.rank_weight = 1e-3
    mod2 = RFMFusion(dim, frame_chunk=2).to(DEV).float()
    mod2.rank_weight = 1e-3
    Z2, rank2 = mod2(XT, XS, b=2)
    check('低秩项为标量且有限', rank2 is not None and rank2.dim() == 0 and torch.isfinite(rank2),
          'rank=%.4f' % (rank2.item() if rank2 is not None else float('nan')))
    mod2.zero_grad()
    (Z2.sum() + rank2).backward()
    check('低秩项可反传', all(p.grad is None or torch.isfinite(p.grad).all()
                             for p in mod2.parameters()))


def test_pointwise():
    print('\n[3] A1 PointwiseFusion')
    torch.manual_seed(0)
    dim, N, L = 1152, 2, 17
    mod = PointwiseFusion(dim).to(DEV).float()
    XT = torch.randn(N, L, dim, device=DEV)
    XS = torch.randn(N, L, dim, device=DEV)
    Z = mod(XT, XS)
    check('初始严格等于 XT', torch.allclose(Z, XT, atol=1e-5),
          'max|diff|=%.2e' % (Z - XT).abs().max().item())
    mod.zero_grad()
    Z.sum().backward()
    check('梯度有限', all(p.grad is not None and torch.isfinite(p.grad).all()
                          for p in mod.parameters()))


def test_align():
    print('\n[4] 坐标对齐')
    B, T, D, H, W = 1, 3, 2, 3, 4
    N = H * W
    # 通道 0 = x, 通道 1 = y（每格值等于自身坐标）
    xs = torch.arange(W).view(1, 1, 1, W).expand(B, T, H, W).float()
    ys = torch.arange(H).view(1, 1, H, 1).expand(B, T, H, W).float()
    S5 = torch.zeros(B, T, D, H, W)
    S5[:, :, 0] = xs
    S5[:, :, 1] = ys

    # 初始坐标（行优先）
    gy, gx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    pos0 = torch.stack([gx.flatten(), gy.flatten()], -1).float()        # [N,2]
    pos = pos0.view(1, N, 1, 2).repeat(1, 1, T, 1)                      # [B,N,T,2]
    reset = torch.zeros(1, N, T, 1)

    # slot / 静态 nte_center 都应等于行优先展开
    slot = align_spatial_features(S5, pos, reset, mode='slot')
    ref = S5.permute(0, 1, 3, 4, 2).reshape(B * T, N, D)
    check('slot = 行优先展开', torch.allclose(slot, ref))
    nte = align_spatial_features(S5, pos, reset, mode='nte_center')
    check('静态坐标采样=原特征', torch.allclose(nte, ref, atol=1e-5),
          'max|diff|=%.2e' % (nte - ref).abs().max().item())

    # 移动跟踪：把槽位 5 在 t=1 的坐标设为 (2.5, 1.0)，普通跟踪(t=1)应读上一帧坐标=(槽位原坐标)
    # 然后再在 t=2 改为普通跟踪，此时读到 t=1 保存的新坐标 -> (2.5,1.0)
    pos2 = pos.clone()
    pos2[0, 5, 1, 0] = 2.5
    pos2[0, 5, 2, 0] = 2.4
    nte2 = align_spatial_features(S5, pos2, reset, mode='nte_center')
    # t=1 普通跟踪读 pos[t-1]=(槽位5原坐标 x=1,y=1)
    check('t=1 普通跟踪用旧坐标', abs(nte2[1, 5, 0].item() - 1.0) < 1e-5, 'x=%.3f' % nte2[1, 5, 0])
    # t=2 普通跟踪读 pos[t-1]=(t=1 的新坐标 2.5,1.0)
    check('t=2 普通跟踪用上一帧新坐标', abs(nte2[2, 5, 0].item() - 2.5) < 1e-5, 'x=%.3f' % nte2[2, 5, 0])

    # ATD 重置：t=1 重置时用本帧新坐标
    rst1 = reset.clone()
    rst1[0, 5, 1, 0] = 1.0
    nte3 = align_spatial_features(S5, pos2, rst1, mode='nte_center')
    check('ATD 重置用本帧新坐标', abs(nte3[1, 5, 0].item() - 2.5) < 1e-5, 'x=%.3f' % nte3[1, 5, 0])


def count_params():
    print('\n[5] 参数量')
    dim = 1152
    scan = GuidedScan(dim)
    rfm = RFMFusion(dim)
    conv = PointwiseFusion(dim)
    for name, m in [('GuidedScan', scan), ('RFM', rfm), ('Pointwise(A1)', conv)]:
        n = sum(p.numel() for p in m.parameters())
        print('  %-14s %.3f M' % (name, n / 1e6))


if __name__ == '__main__':
    print('device =', DEV)
    test_guided_scan()
    test_rfm()
    test_pointwise()
    test_align()
    count_params()
    print('\n==== %s ====' % ('ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)))
    sys.exit(0 if not FAIL else 1)
