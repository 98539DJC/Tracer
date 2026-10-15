import torch

CKPT = 'output/MSVD/checkpoints/my_run_qlora_v3/checkpoint-200/non_lora_trainables.bin'

raw = torch.load(CKPT, map_location='cpu')
print(f'total keys: {len(raw)}')
print('--- raw key samples (first/last) ---')
ks = list(raw.keys())
for k in ks[:4] + ks[-4:]:
    print('  ', k)

print('\n--- projector weight status (did optimizer actually step?) ---')
def stat(name):
    found = 0
    for k, v in raw.items():
        if name in k:
            found += 1
            print(f'  {k}: shape={tuple(v.shape)} max_abs={v.abs().max().item():.3e} mean_abs={v.abs().mean().item():.3e}')
    if found == 0:
        print(f'  (no key contains "{name}")')

# W_sim 零初始化 -> 非零说明 optimizer 真的 step 了
stat('W_sim')
# W_pos normal_(std=1e-3) 初始化
stat('W_pos')
# W_Q trunc_normal(0.02) 初始化
stat('W_Q')

print('\n--- remapped keys (what gen_chunk feeds load_state_dict) ---')
remapped = {k[11:] if k.startswith('base_model.') else k: v for k, v in raw.items()}
if any(k.startswith('model.model.') for k in remapped):
    remapped = {k[6:] if k.startswith('model.') else k: v for k, v in remapped.items()}
rks = list(remapped.keys())
for k in rks[:4] + rks[-4:]:
    print('  ', k)

# 检查 remap 后是否都以 model.mm_projector 开头
bad = [k for k in rks if not k.startswith('model.mm_projector.')]
print(f'\nremapped keys NOT starting with model.mm_projector.: {len(bad)}')
for k in bad[:10]:
    print('  ', k)
