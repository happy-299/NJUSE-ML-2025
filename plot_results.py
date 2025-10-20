from utils.visualization import plot_training_history
import os

# 创建图表保存目录
os.makedirs('./outputs/figures', exist_ok=True)

# 绘制训练历史
plot_training_history('./outputs/example/history.json', save_dir='./outputs/figures')
print("图表已保存到 ./outputs/figures 目录")