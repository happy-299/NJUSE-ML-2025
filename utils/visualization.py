# utils/visualization.py
# 绘图和可视化工具示例

import matplotlib.pyplot as plt


def plot_loss_curve(losses, save_path=None):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
