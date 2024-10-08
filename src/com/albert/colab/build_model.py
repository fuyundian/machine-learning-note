import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import RMSprop


def build_model(my_learning_rate):
    # """创建并编译一个简单的线性回归模型。"""
    # 大多数简单的 tf.keras 模型都是顺序的。
    # 顺序模型包含一层或多层。
    model = tf.keras.models.Sequential([
        # 描述模型的地形。
        # 简单线性回归模型的拓扑
        # 是单层中的单个节点。
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])
    # 将模型地形编译成代码
    # TensorFlow可以高效执行。配置
    # 训练以最小化模型的均方误差。
    optimizer = RMSprop(learning_rate=my_learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def train_model(model, feature, label, epochs, batch_size):
    # """Train the model by feeding it data."""

    # 将特征值和标签值输入
    # 模型。模型将训练指定数量
    # epochs，逐渐学习特征值如何
    # 与标签值相关。
    history = model.fit(x=feature, y=label, batch_size=batch_size, epochs=epochs)

    # 收集训练模型的权重和偏差。
    trained_weight = model.get_weights()[0][0]
    trained_bias = model.get_weights()[1]

    # 纪元列表与
    # 其余的历史。
    epochs = history.epoch

    # 收集每个纪元的历史（快照）。
    hist = pd.DataFrame(history.history)

    # 特别收集模型的均方根
    # 每个时期的平方误差。
    root_mse = hist["root_mean_squared_error"]
    return trained_weight, trained_bias, epochs, root_mse


# @title 定义绘图函数
def plot_the_model(trained_weight, trained_bias, feature, label):
    # ""根据训练特征和标签绘制经过训练的模型。""

    # 标记轴。
    plt.xlabel("feature")
    plt.ylabel("label")
    # 绘制特征值与标签值的图。
    plt.scatter(feature, label)

    # 创建一条代表模型的红线。红线开始
    # 在坐标 (x0, y0) 处并在坐标 (x1, y1) 处结束。
    x0 = 0
    y0 = trained_bias
    x1 = feature[-1]
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], c='r')

    # 渲染散点图和红线。
    plt.show()


def plot_the_loss_curve(epochs, rmse):
    # """绘制损失曲线，显示损失与历元的关系。"""
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min(), rmse.max()])
    plt.show()

