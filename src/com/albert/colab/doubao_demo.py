
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def dou_bao():
    x_train = np.linspace(0, 10, 100)  # 生成 100 个在 0 到 10 之间均匀分布的点
    y_train = 2 * x_train + 1 + np.random.normal(0, 1, size=100)  # 生成对应的 y 值，并添加一些噪声

    # 构建神经网络模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))  # 一个神经元的全连接层，输入维度为 1
    ])

    # 编译模型
    model.compile(optimizer='RMSprop', loss='mse')  # 使用均方误差作为损失函数，adam 优化器

    # 训练模型
    history = model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=0)  # 训练 50 个 epoch，批量大小为 10

    # 预测并绘制拟合曲线
    x_pred = np.linspace(0, 10, 200)  # 生成用于预测的 x 值
    y_pred = model.predict(x_pred)  # 进行预测

    # 绘制训练过程中的损失变化
    plt.plot(history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    # 绘制原始数据和拟合曲线
    plt.scatter(x_train, y_train, label='Original Data')
    plt.plot(x_pred, y_pred, color='red', label='Fitted Line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()