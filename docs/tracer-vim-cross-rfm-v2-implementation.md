# Tracer + Vim Cross-Mamba + RFM（方案 v2）实现记录

> 依据：`Tracer-Vim-CrossMamba-RFM-Server-Implementation-v2.pdf`（2026-11-11）
> 基础论文：Jing et al., *M³amba: Multi-Modality Mamba for Image Fusion*, IEEE TIP 35, 2026
> 本地代码：分支 `vim-spatial-v1`（未提交）

## 1. 实现进度

| 模块 | 文件 | 状态 |
|---|---|---|
| GuidedScan（Cross-Mamba 单向） | `videollama2/model/projector.py` | ✅ 已实现，与逐时间步参考递推一致（max diff 4.7e-10） |
| RFM（共享 softmax 关系融合） | 同上 | ✅ 形状 / 初始近恒等 / 梯度 / 低秩项自检通过 |
| A1 PointwiseFusion（2304→1152） | 同上 | ✅ 初始严格等于 XT |
| 坐标对齐（NTE 实际采样中心 / slot） | `align_spatial_features` | ✅ 静态/移动/ATD 重置断言通过 |
| 连接器 `SpatioTemporalCrossConnector` | 同上；`build_vision_projector` 分发 `tracer_vim_cross` | ✅ 已接入 |
| config 字段传递 | `videollama2_arch.py` + `train.py` ModelArguments | ✅ `st_fusion_mode/st_scan_mode/st_align_mode/st_fusion_scale/st_rank_weight/st_frame_chunk` |
| `trainable_scope=st`（只训新模块） | `videollama2_trainer.py`/`train.py` | ✅ |
| A 组只反传 caption loss；rank 项可选 | `videollama2_trainer.py` | ✅ |
| 训练脚本 | `finetune_qlora_MSVD_cross.sh` | ✅ |
| 首轮驱动 A1–A5 | `run_st_cross_round1.sh` | ✅ |
| 自检工具 | `tools/verify_cross_rfm.py` | ✅ ALL PASS |
| 2 步端到端冒烟（A4） | — | ✅ PASS（loss 有限、max_alloc 30.56GB/卡、`st_delta_ratio≈0.0028`） |
| 真实 NTE 坐标打印 | — | ✅ 1 步冒烟：t=0 初始网格；t=1 reset=0 取旧坐标；t=2 reset=1 取新坐标 |
| 整包投影器保存 | `train.py` / `videollama2_trainer.py` | ✅ scope 加 `'st'`，checkpoint 保存 394 tensors（B0 时间分支 + 79 新参数） |
| 评测读取 st 字段 | `gen_chunk.py` | ✅ 否则默认 `st_fusion_mode=off` 会静默退化成 B0 |

## 2. 数据流（与 PDF §9.1 一致）

```
X  = pre_norm(patch features)                         [B,16,27,27,1152]
T5, pos_fwd, i_fwd = temporal_branch(X)               BCM 残差后 [B,16,1152,27,27]
S5 = X + 0.1 * Vim(RMSNorm(X))                        逐帧 BiMamba v2 [B,16,1152,27,27]
Sa = align(S5, pos_fwd, i_fwd, align_mode)            [16B,729,1152]
XT = T; XS = Sa
if scan=off:    XT,XS 不变
if scan=self:   XT = T + M_ts(T,T);   XS = Sa + M_st(Sa,Sa)
if scan=cross:  XT = T + M_ts(T,Sa);  XS = Sa + M_st(Sa,T)
if fusion=conv: Z = pointwise(cat(XT,XS))
if fusion=rfm : Z = RFM(XT,XS)          # 自读 P_m + 交叉读共享 P
F = T + γ(Z−T)                          γ=0.1 固定
visual_tokens = s1/sampler/s2/readout(F)   [B,1352,3584]
```

- 坐标语义（用户确认）：首帧用初始坐标；普通跟踪用 `pos_fwd[t-1]`；ATD 重置（`i_fwd[t]=1`）用 `pos_fwd[t]`；全部在当前帧空间图上取样。
- `slot` 模式不做采样，S 按初始槽位行优先展开（A5）。

## 3. 关键口径（用户 2026-11-11 确认）

1. **RFM 自读**用本路 `P_m`，交叉读用共享 `P=softmax((E_T+E_S)/2)`。
2. **LN 位置**：`LN(自读输出) + 交叉读输出`（M³amba 式 4 顺序）。
3. **扫描方向**：Vim 保留正反向；Cross-Mamba 每帧两次、同一行优先顺序、角色互换。方向数留作独立变量。
4. **低秩分母**：`B·F·4·2·L`；`10^-3` 只是试跑值，A1–A5 一律 `rank=0`。
5. **张量形状**：融合内部 `[16B,729,1152]`；`F` 先 reshape `[B,16,27,27,1152]` 再 permute `[B,16,1152,27,27]` 交回原投影器尾部。
6. **NTE 坐标**：以运行中的 `pos_fwd/i_fwd` 为准（服务器首步打印真实样本核对）。
7. **A1** 只有一个 `2304→1152` 线性层；`[I,0]` 初始化；`γ=0` 应严格等价 B0。
8. **损失**：A1–A5 只反传加权 caption loss；`cor/cyc/div` 仅日志。低秩组另加 `λ·L_rank`。
9. **训练配方**：首轮 A 组统一 2 卡 × bs3 × acc10。B0@250 为固定起点。
   ⚠️ caption 是未归一化 wordweight sum，DeepSpeed 在 backward 内除以 `gradient_accumulation_steps`；
   与 B0 对比时必须保持同 micro-batch/acc（或显式恢复 `6/Bµ` 缩放）。
10. **结论口径**：先短跑 A1 的坐标/同编号对照，再 A2；互引导 A4 vs A3；对齐 A4 vs A5。验证集选点后 test。

## 4. 初始化与两档训练协议

| 档位 | JOINT | 初始化 | 可训练 | 损失 | 步数 |
|---|---|---|---|---|---|
| A 组（§6.1） | 0 | B0@250（时间投影器 + LoRA） | 仅新模块（79 参数） | caption only | 首轮 200 |
| 从头联合 | 1 | Base `mm_projector.bin` + 全新 LoRA | 时间分支 + Vim + Cross-Mamba + RFM + LoRA | caption + cor/cyc/div | 812 |

- 用户 2026-11-11 决定：**完整结构 A4 先按「从头时间空间联合训练」跑**（不是 §6.1 的冻结时间分支）。
- 联合档的 NTE/BCM 倍率用 ×1.0（与 C-Joint 一致，方案 §8.1）。
- 两档都使用 2 卡 × bs3 × acc10、seed42、feat32、16 帧、4096。

- 新模块：同一 `seed=42` 下按固定顺序构建；Vim `out_proj` 与扫描 `out_proj` 用 std 1e-3 非零初始化；
  RFM `pointwise_skip = [I,0]`；`pointwise_top`/`ffn2` std 1e-3。
- 冻结档 `trainable_scope=st`：冻结 SigLIP / 共享 pre_norm / 时间分支 / 原投影器 / LLM / LoRA；
  只训 `sp_norm, sp_mamba, scan_ts, scan_st, fuse`。

## 5. 运行

```bash
# 自检
CUDA_VISIBLE_DEVICES=3 python -u tools/verify_cross_rfm.py

# 2 步冒烟（A4，最复杂）
RUN_NAME=a4_smoke MAX_STEPS=2 ST_FUSION_MODE=rfm ST_SCAN_MODE=cross \
  ST_ALIGN_MODE=nte_center bash finetune_qlora_MSVD_cross.sh

# 首轮 A1–A5（冻结 B0 时间分支，200 步 + valid 扫描）
ARMS="a1 a2" bash run_st_cross_round1.sh
FULL=1 bash run_st_cross_round1.sh        # 812 步

# 完整结构 A4：从头时间+空间+融合+LoRA 联合训练（812 步，先 2 步冒烟）
bash run_st_cross_joint.sh
```

输出：
- checkpoint `output/MSVD/checkpoints/st_<arm>_s42/`
- 日志 `output/MSVD/logs/st_<arm>_s42_*.log`
- valid 曲线 `output/MSVD/st_<arm>_s42_all_ckpt_summary_valid.tsv`

## 6. 待办 / 风险

- [x] 2 步冒烟：`[ST]`/`[SCOPE]`/`[PROJ] loaded 315/394`/`[CREATE_OPTIM] req_grad=79`、loss 有限、无 OOM。
- [x] checkpoint 保存整包投影器（394 tensors），`gen_chunk.py` 会恢复 st_* 开关。
- [ ] `γ=0` 与 B0@250 的 forward 等价性（代码路径已直接 early-return，待端到端 token 对比）。
- [x] 服务器首步打印真实样本的旧坐标/新坐标/ATD 标记/实际采样中心。
- [ ] RFM 显存与耗时（L=729，头数 4）；必要时 `ST_RFM_CKPT=1` 或减小 `st_frame_chunk`。
- [ ] 与 B0 同 micro-batch 的因果对照（见 §3.9）。
- [ ] 首轮 A1/A1s 训练中（200 步 + valid 扫描），随后 A2 → A3/A4/A5。

## 7. A4 联合训练中断与 ckpt 评测结果（2026-11-12）

- **训练中断**：`st_a4_joint_s42` 在 step 254/812（01:28）被外部 SIGKILL（无 traceback/OOM/重启，wrapper 无 exit 行）。`checkpoint-250` 完整（含 zero2 优化器状态），`train.py` 有 auto-resume，可续训。
- **评测口径**（用户指定）：test 集 + `siglip_feat16` + σ=3.7，`EVAL_GPUS=3` 单卡，`EVAL_SPLIT=test EVAL_FEATURE_DIR=datasets/MSVD_sft/siglip_feat16`。
- **结果** `output/MSVD/st_a4_joint_s42_all_ckpt_summary_test.tsv`：

| step | 50 | 100 | 150 | 200 | 250 |
|---|---|---|---|---|---|
| A4 joint CIDEr(×100) | 186.43 | 184.29 | 182.67 | 183.29 | 178.77 |
| B0-lrhalf CIDEr(×100) | 187.91 | 184.84 | 186.29 | 188.26 | 184.98 |

- **结论**：每个 step 均低于 B0-lrhalf 同口径，且从 step50 起持续下滑 → 从头联合训练 + Cross-Mamba/RFM **无增益、过拟合更快**。不建议续训至 812（趋势只会更差）。

## 8. A4 减半学习率复跑（2026-11-12 起）

- **动机**：A4 全 LR 的 test 曲线峰值出现在第一个测试点（step50=186.43），无法确定真实峰值，怀疑 1e-5/1e-4 对从头联合训练偏高。
- **改动**：`finetune_qlora_MSVD_cross.sh` 原先硬编码 `--learning_rate 1e-5` / `--mm_projector_lr 1e-4`，现改为可覆盖的 `LORA_LR` / `PROJ_LR`（默认值不变）。`NTE_BCM_LR_MULT` 跟随投影器 LR，因此 NTE/BCM 也同步减半。`run_st_cross_joint.sh` 支持 `RUN/MAX_STEPS/SAVE_STEPS/LORA_LR/PROJ_LR/EVAL_CKPTS/EVAL_SPLIT`。
- **本次配方**（其余与 A4 完全一致，单变量对照）：
  - `LORA_LR=5e-6  PROJ_LR=5e-5`（NTE/BCM ×1.0）
  - `RUN=st_a4_joint_lrhalf_s42`，`MAX_STEPS=812`（保持原 cosine 日程），`SAVE_STEPS=25`
  - 评测：`EVAL_SPLIT=test`、feat16、σ=3.7，加密早期点 `25 50 75 100 125 150 200 250 300`（训练结束后自动 sweep；若中途停训则手动跑 `eval_all_checkpoints.sh`）
- **启动**：2026-11-12 18:36 冒烟（exit=0，18:42 全量起跑），GPU 2/3，2 卡 × bs3 × acc10；日志确认 `LoRA=5e-6  projector=5e-5  NTE/BCM x1.0`、`save=25`。
- **首次全量被 SIGKILL（18:51，step 5）**：`/var/log/syslog` 同一时刻出现 GPU 2/3 的 `NVRM: Xid 13 (Out Of Range Address)` + `Xid 43`，连同时的 `nvidia-smi` 进程也被报 Xid 13；cgroup 无 OOM、ECC/retired pages 全 0。GPU 2/3 随后 bf16 matmul 自检通过，判断为驱动/瞬时故障（非 LR 引起，也与 01:28 那次无 Xid 的 SIGKILL 不同）。
- **重启**：18:54 起改用 `systemd-run --user --unit=a4-lrhalf-s42`（脱离终端 session，避免 session 回收连带杀进程），冒烟→全量重跑，LR 参数已在命令行核实。
- **状态**：⏳ 冒烟 → 全量训练中；结果回来后补表格。
- **19:04 全量 OOM（非本 run 泄漏）**：同账号（megaium）另有两个 `/mnt/lzz/VideoRefer-main` 任务占用 GPU2/3——`msvd_gbs60_1gpu_20261112_190140`（PID 1172607，长跑，常驻 ~37.7GB 在 GPU3）和 `ddp_gloo_20261112_190344`（PID 1176430/1176431，2 卡 3 步 smoke，~39GB/卡）。我们的 checkpoint 读取时 GPU3 已被占 76.9GB，故 rank1 OOM。3 步 smoke 于 19:05 结束后 GPU2 全空、GPU3 剩 37.7GB。
- **19:06 二次重启**：再次 `systemd-run --user --unit=a4-lrhalf-s42`，起跑后 GPU2≈38GB、GPU3≈75GB（37.7 对方 + 37.3 我方），可共存但 GPU3 余量紧张。若再 OOM，需等对方任务结束或改单卡。
- **19:08 用户决定暂停，等师兄的 GPU 任务跑完再续**：已 `systemctl --user stop a4-lrhalf-s42`（unit inactive，无残留训练进程），GPU2/3 上现有显存均为 `/mnt/lzz` 任务（PID 1172607、1186539/1186540）。本次 lrhalf 只完成冒烟，全量未存过 checkpoint（<25 步），**之后需要整跑重来**（不存在可续的 ckpt）。脚本改动（LORA_LR/PROJ_LR 参数化、run_st_cross_joint.sh 可覆盖项）保留。
- **恢复方法**：等 GPU2/3 空出后
  `systemd-run --user --unit=a4-lrhalf-s42 --collect --working-directory=/home/megaium/Tracer-main --setenv=LORA_LR=5e-6 --setenv=PROJ_LR=5e-5 --setenv=RUN=st_a4_joint_lrhalf_s42 --setenv=SAVE_STEPS=25 --setenv=MAX_STEPS=812 --setenv=EVAL_SPLIT=test --setenv="EVAL_CKPTS=25 50 75 100 125 150 200 250 300" --setenv=GPUS=2,3 /bin/bash /home/megaium/Tracer-main/run_st_cross_joint.sh`
