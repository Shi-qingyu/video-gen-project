import matplotlib.pyplot as plt

# 数据：方法名称、All列的 EF 和 MF 值
methods = [
    "MotionDirector", "SMA", "MotionClone", "MOFT", "DMT (CogVideoX)",
    "DreamBooth (CogVideoX)", "MotionInversion (CogVideoX)",
    "Ours (CogVideoX)", "Ours (HunyuanVideo)", "Ours (Step-Video-T2V)"
]
ef_values = [31.9, 31.6, 30.8, 33.0, 30.5, 28.4, 26.6, 31.2, 31.9, 31.4]
mf_values = [67.7, 55.1, 78.9, 52.5, 65.1, 80.4, 85.0, 85.8, 85.9, 85.8]

# 分离“我们的”方法和其他方法
others_mf, others_ef, others_methods = [], [], []
ours_mf, ours_ef, ours_methods = [], [], []
for i, method in enumerate(methods):
    if "Ours" in method:
        ours_mf.append(mf_values[i])
        ours_ef.append(ef_values[i])
        ours_methods.append(method)
    else:
        others_mf.append(mf_values[i])
        others_ef.append(ef_values[i])
        others_methods.append(method)

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制其他方法的散点图（蓝色圆点）
ax.scatter(others_mf, others_ef, s=100, color='dodgerblue', edgecolor='black')
for i, method in enumerate(others_methods):
    ax.annotate(method, (others_mf[i], others_ef[i]), textcoords="offset points",
                xytext=(5, 5), fontsize=10)

# 绘制我们的方法的散点图（红色星形点，高亮显示）
ax.scatter(ours_mf, ours_ef, s=150, color='red', marker='*', edgecolor='black')
for i, method in enumerate(ours_methods):
    ax.annotate(method, (ours_mf[i], ours_ef[i]), textcoords="offset points",
                xytext=(5, 5), fontsize=10, color='red')

# 设置坐标轴标签和标题
ax.set_xlabel("Motion Fidelity (MF)", fontsize=14)
ax.set_ylabel("Edit Fidelity (EF)", fontsize=14)
ax.set_title("Edit Fidelity vs Motion Fidelity", fontsize=16)

# 自动调整布局并保存图片
plt.tight_layout()
plt.grid()
plt.savefig("scatter.jpg")
