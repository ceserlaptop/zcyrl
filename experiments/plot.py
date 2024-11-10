import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from matplotlib import rcParams

scenario_name = "coverage_0"

fileOrder = -2
data_path = "./" + scenario_name + "/" + os.listdir("./" + scenario_name)[fileOrder] + "/plots/"
figure_save_path = "./" + scenario_name + "/" + os.listdir("./" + scenario_name)[fileOrder] + "/figures/"

step = 100

# 设置图像字体
config = {
    "font.family": 'Times New Roman',  # 设置字体类型
}
rcParams.update(config)

if __name__ == '__main__':
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建

    reward_data = pd.read_csv(os.path.join(data_path, 'train_rewards.csv'), sep=',', header=None, usecols=[0])
    collision_data = pd.read_csv(os.path.join(data_path, 'collision.csv'), sep=',', header=None, usecols=[0])
    coverage_rate_data = pd.read_csv(os.path.join(data_path, 'train_coverage_rate.csv'), sep=',', header=None, usecols=[0])
    outRange_data = pd.read_csv(os.path.join(data_path, 'outRange.csv'), sep=',', header=None, usecols=[0])
    done_steps_data = pd.read_csv(os.path.join(data_path, 'modify_done_steps.csv'), sep=',', header=None, usecols=[0])
    test_steps_data = pd.read_csv(os.path.join(data_path, 'test_done_steps.csv'), sep=',', header=None, usecols=[0])

    mean_rew, mean_col, mean_cov, mean_done, mean_out = [], [], [], [], []

    for i in range(0, len(reward_data), step):
        mean_rew = np.append(mean_rew, np.mean(reward_data[i:i + step]))
        mean_col = np.append(mean_col, np.mean(collision_data[i:i + step]))
        mean_cov = np.append(mean_cov, np.mean(coverage_rate_data[i:i + step]))
        mean_out = np.append(mean_out, np.mean(outRange_data[i:i + step]))
        mean_done = np.append(mean_done, np.mean(done_steps_data[i:i + step]))

    palette = plt.get_cmap('Set1')
    # 绘制奖励函数曲线
    plt.figure()
    plt.plot(mean_rew, color=palette(1))
    plt.title("Mean Episode Reward")
    plt.xlabel("episodes/%d" % step)
    plt.ylabel("reward")
    plt.grid()
    plt.savefig(os.path.join(figure_save_path, 'reward.jpg'), dpi=600)

    # 绘制覆盖率
    plt.figure()
    plt.plot(mean_cov, color=palette(1))
    plt.title("Mean Coverage Rate")
    plt.xlabel("episodes/%d" % step)
    plt.ylabel("Rate")
    plt.grid()
    plt.ylim([0.0, 1.1])
    plt.savefig(os.path.join(figure_save_path, 'Coverage.jpg'), dpi=600)

    # 绘制碰撞率
    plt.figure()
    plt.plot(mean_col, color=palette(1))
    plt.title("Mean Collision Rate")
    plt.xlabel("episodes/%d" % step)
    plt.ylabel("Rate")
    plt.grid()
    plt.ylim([0.0, 1.1])
    plt.savefig(os.path.join(figure_save_path, 'Collision.jpg'), dpi=600)

    # 绘制碰撞率
    plt.figure()
    plt.plot(mean_col, color=palette(1))
    plt.title("Mean Out Range Rate")
    plt.xlabel("episodes/%d" % step)
    plt.ylabel("Rate")
    plt.grid()
    plt.ylim([0.0, 1.1])
    plt.savefig(os.path.join(figure_save_path, 'outRange.jpg'), dpi=600)

    # 绘制done_steps
    plt.figure()
    plt.plot(mean_done, color=palette(1))
    plt.title("Mean Done Steps")
    plt.xlabel("episodes/%d" % step)
    plt.ylabel("Rate")
    plt.grid()
    plt.savefig(os.path.join(figure_save_path, 'done.jpg'), dpi=600)

    plt.figure()
    plt.plot(test_steps_data, color=palette(1))
    plt.title("Mean Test Done Steps")
    plt.xlabel("episodes/%d" % step)
    plt.ylabel("Rate")
    plt.grid()
    plt.savefig(os.path.join(figure_save_path, 'test_done.jpg'), dpi=600)