from matplotlib import pyplot as plt

# 创建绘制实时损失的动态窗口
plt.ion()


def showVertices(points):
    plt.clf()
    plt.plot(points[:, 0], points[:, 1])  # 画出当前x列表和y列表中的值的图形
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.pause(0.1)
    plt.ioff()

