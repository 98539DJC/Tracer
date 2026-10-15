"""实测：构建 TracerMambaConnector，加载 mm_projector.bin，检查 readout/s1 是否真的加载上。

用法:
    python check_projector_load.py [mm_projector.bin路径]
"""
import torch, sys, os, glob
from types import SimpleNamespace

def find_bin():
    patterns = [
        os.path.expanduser("~/.cache/huggingface/hub/**/mm_projector.bin"),
    ]
    found = []
    for p in patterns:
        found += glob.glob(p, recursive=True)
    return found

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        found = find_bin()
        path = found[0] if found else None
    if path is None:
        print("找不到 bin，手动传: python check_projector_load.py <bin路径>")
        return
    print(f"bin: {path}")

    # 构造 config（关键字段）
    config = SimpleNamespace(
        mm_hidden_size=1152,   # SigLIP 输出通道
        hidden_size=3584,      # Qwen2-7B 隐层
        mm_projector_type='route_mamba',
    )

    from videollama2.model.projector import TracerMambaConnector
    proj = TracerMambaConnector(config)

    # 加载 bin + 模拟 get_w 提取
    bin_w = torch.load(path, map_location='cpu')
    get_w = lambda weights, kw: {k.split(kw + '.')[1]: v for k, v in weights.items() if kw in k}
    mm_w = get_w(bin_w, 'mm_projector')

    # 记录 readout / s1 加载前权重
    readout_before = {k: v.clone() for k, v in proj.readout.state_dict().items()}
    s1_before = {k: v.clone() for k, v in proj.s1.state_dict().items()}

    # 执行加载（跟 videollama2_arch.py:98 一样）
    result = proj.load_state_dict(mm_w, strict=False)

    print(f"\n=== load_state_dict 结果 ===")
    print(f"missing_keys ({len(result.missing_keys)}): {result.missing_keys[:10]}")
    print(f"unexpected_keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:10]}")

    print(f"\n=== readout 是否加载（随机→bin 值 = 已加载）===")
    readout_after = proj.readout.state_dict()
    for k in readout_before:
        changed = not torch.equal(readout_before[k], readout_after[k])
        print(f"  {k}: {'已加载 ✓' if changed else '未加载 ✗（还是随机）'}")

    print(f"\n=== s1 是否加载（抽查前几个 key）===")
    s1_after = proj.s1.state_dict()
    cnt_loaded = 0
    cnt_total = 0
    for k in s1_before:
        cnt_total += 1
        changed = not torch.equal(s1_before[k], s1_after[k])
        if changed:
            cnt_loaded += 1
    print(f"  s1: {cnt_loaded}/{cnt_total} 个 key 加载上")
    for k in list(s1_before)[:5]:
        changed = not torch.equal(s1_before[k], s1_after[k])
        print(f"    {k}: {'已加载' if changed else '未加载'}")

if __name__ == '__main__':
    main()
