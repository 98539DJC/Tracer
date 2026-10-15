"""检查 mm_projector.bin 的 key 格式，验证 readout/s1 权重能否真正加载上。

用法:
    python check_mm_projector_load.py [mm_projector.bin路径]

不给路径则自动搜 HF cache。
关键看两点:
  1. bin 的 key 前缀是 'mm_projector.' 还是 'model.mm_projector.' / 别的
  2. get_w('mm_projector') 提取后 readout 的 shape 是否 = (3584, 3584)
"""
import torch, sys, os, glob

def find_bin():
    patterns = [
        os.path.expanduser("~/.cache/huggingface/hub/**/mm_projector.bin"),
        os.path.expanduser("~/.cache/huggingface/**/mm_projector.bin"),
    ]
    found = []
    for p in patterns:
        found += glob.glob(p, recursive=True)
    return found

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        found = find_bin()
        print("未提供路径，自动搜索 HF cache 找到:")
        for f in found:
            print("  ", f)
        if not found:
            print("  没找到。手动传路径: python check_mm_projector_load.py <bin路径>")
            return
        path = found[0]
        print(f"\n使用: {path}")

    bin_w = torch.load(path, map_location='cpu')
    print(f"\n=== bin 总 key 数: {len(bin_w)} ===")
    for k, v in bin_w.items():
        print(f"  {k}: {tuple(v.shape)}")

    # 模拟 videollama2_arch.py:93-98 的 get_w
    get_w = lambda weights, kw: {k.split(kw + '.')[1]: v for k, v in weights.items() if kw in k}
    mm_w = get_w(bin_w, 'mm_projector')
    print(f"\n=== get_w('mm_projector') 提取到 {len(mm_w)} 个 key ===")
    for k, v in mm_w.items():
        print(f"  {k}: {tuple(v.shape)}")

    # 检查关键模块的 key 和 shape
    for name in ['readout', 's1', 's2', 'sampler']:
        keys = [k for k in mm_w if k.startswith(name + '.')]
        print(f"\n[{name}] {len(keys)} 个 key")
        for k in keys:
            print(f"  {k}: {tuple(mm_w[k].shape)}")

    print("\n=== 期望对照（TracerMambaConnector 当前结构）===")
    print("  readout = build_mlp(2, 3584, 3584): readout.0.weight (3584,3584), readout.2.weight (3584,3584)")
    print("  s1 = RegStage(4, 1152, 3584): 含多层卷积 key（s1.stages.xxx 或类似）")
    print("  nte / m1(BCM): 随机初始化，bin 里本就不该有")

if __name__ == '__main__':
    main()
