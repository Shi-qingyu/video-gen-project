from src.pipeline import MyStepWiseCogVideoXPipeline, MyTextToVideoSDPipeline
import torch
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os

# 创建保存目录
os.makedirs('plots', exist_ok=True)
prompt = "A woman is dancing on the ground."
generator = torch.Generator(device="cuda").manual_seed(0)

try:
    # 加载第一种 pipeline
    pipe_cog = MyStepWiseCogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # 执行 pipeline 并获取损失
    loss_tracker_cog = pipe_cog(
        prompt,
        generator=generator,
    )

    # 确认损失数据的结构
    print("CogVideoX Loss Data:", loss_tracker_cog)

    # 转换为 NumPy 数组
    data_cog = np.array(loss_tracker_cog)
    x_cog = range(len(data_cog))

    # 创建图形对象并设置大小
    plt.figure(figsize=(10, 6))

    # 绘制第一条折线
    plt.plot(x_cog, data_cog, marker='x', linestyle='-', label='CogVideoX Loss')

except Exception as e:
    print(f"CogVideoX 处理时发生错误: {e}")
finally:
    # 清理资源
    del pipe_cog
    gc.collect()
    torch.cuda.empty_cache()

try:
    # 加载第二种 pipeline
    pipe_sd = MyTextToVideoSDPipeline.from_pretrained(
        "cerspense/zeroscope_v2_576w",
        torch_dtype=torch.float16
    )
    pipe_sd.enable_model_cpu_offload()

    # 执行 pipeline 并获取损失
    loss_tracker_sd = pipe_sd(
        prompt,
        num_inference_steps=50,
        height=320,
        width=576,
        num_frames=24,
        generator=generator,
    )

    # 确认损失数据的结构
    print("TextToVideoSD Loss Data:", loss_tracker_sd)

    # 转换为 NumPy 数组
    data_sd = np.array(loss_tracker_sd)
    x_sd = range(len(data_sd))

    # 绘制第二条折线
    plt.plot(x_sd, data_sd, marker='o', linestyle='-', label='ZeroScope Loss')

except Exception as e:
    print(f"TextToVideoSD 处理时发生错误: {e}")

# 添加标题和轴标签
plt.title('CogVideoX-5b versus ZeroScope')
plt.xlabel('Step')
plt.ylabel('loss')

# 添加网格和图例
plt.grid(True)
plt.legend()

# 保存图像
plt.savefig('plots/loss.png', dpi=300)

# 如果需要显示图形，可以取消注释以下行
# plt.show()