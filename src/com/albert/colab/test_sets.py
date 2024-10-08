import os

import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import RMSprop


# 创建模型
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


def train_model(model, training_df, feature, label, epochs, batch_size):
    # 将特征值和标签值输入
    # 模型。模型将训练指定数量
    # epochs，逐渐学习特征值如何
    # 与标签值相关。
    history = model.fit(x=training_df[feature], y=training_df[label], batch_size=batch_size, epochs=epochs)

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
def plot_the_model(trained_weight, training_df, trained_bias, feature, label):
    # 标记轴。
    plt.xlabel(feature)
    plt.ylabel(label)
    # 根据数据集的 200 个随机点创建散点图。
    random_examples = training_df.sample(n=200)
    plt.scatter(random_examples[feature], random_examples[label])

    # 创建一条代表模型的红线。红线开始
    # 在坐标 (x0, y0) 处并在坐标 (x1, y1) 处结束。
    x0 = 0
    y0 = trained_bias
    x1 = random_examples[feature].max()
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
    plt.ylim([rmse.min() * 0.97, rmse.max()])
    plt.show()


def read_csv():
    pd.options.display.max_rows = 10
    pd.options.display.float_format = "{:.1f}".format
    # 获取文件目录
    curPath = os.path.abspath(os.path.dirname(__file__))
    # 获取项目根路径，内容为当前项目的名字
    rootPath = curPath[:curPath.find('machine-learning-note') + len('machine-learning-note')]
    # # 构建 CSV 文件的完整路径
    file_path = os.path.join(rootPath, 'src/resouces/california_housing_train.csv')
    # # 读取 CSV 文件
    training_df = pd.read_csv(file_path)
    training_df["median_house_value"] /= 1000.0

    # Print the first rows of the pandas DataFrame.
    training_df.head()
    training_df.describe()
    return training_df


def predict_house_values(n, training_df, my_model, feature, label):
    """Predict house values based on a feature."""

    batch = training_df[feature][10000:10000 + n]
    predicted_values = my_model.predict_on_batch(x=batch)

    print("feature   label          predicted")
    print("  value   value          value")
    print("          in thousand$   in thousand$")
    print("--------------------------------------")
    for i in range(n):
        print("%5.0f %6.0f %15.0f" % (
        training_df[feature][10000 + i], training_df[label][10000 + i], predicted_values[i][0]))
