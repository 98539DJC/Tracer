from fvcore.nn import FlopCountAnalysis, parameter_count



def analyze_model_param(model, inputs):
    """
    分析复杂 forward 函数的模型
    """
    
    # 创建 flop counter
    flops_counter = FlopCountAnalysis(model, inputs)
    
    # 打印每层 FLOPs
    print("\nFLOPs by layer:")
    for module_name, module_flops in flops_counter.by_module().items():
        if module_flops > 0:  # 只显示有 FLOPs 的层
            print(f"{module_name:50} {module_flops / 1e9:>10.2f} G")
    
    total_flops = flops_counter.total()
    print(f"\nTotal FLOPs: {total_flops / 1e9:.2f} G")
    
    return  total_flops
