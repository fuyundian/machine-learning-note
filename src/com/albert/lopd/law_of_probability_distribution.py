import numpy as np
from matplotlib import pyplot as plt


def lopd_demo():
    mu = 0  # 均值
    sigma = 1  # 标准差
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(1 - (x - mu) ** 2 / (2 * sigma ** 2))

    # 绘制图表
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel("Probability Density")
    plt.title("normal distribution")
    plt.grid(True)
    plt.show()
