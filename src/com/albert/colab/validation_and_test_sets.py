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


def train_model(model, df, feature, label, my_epochs,
                my_batch_size=None, my_validation_split=0.1):

    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=my_batch_size,
                        epochs=my_epochs,
                        validation_split=my_validation_split)

    # Gather the model's trained weight and bias.
    trained_weight = model.get_weights()[0][0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the
    # rest of history.
    epochs = history.epoch

    # Isolate the root mean squared error for each epoch.
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return epochs, rmse, history.history


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


def plot_the_loss_curve(epochs, mae_training, mae_validation):
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
    plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
    plt.legend()

    # We're not going to plot the first epoch, since the loss on the first epoch
    # is often substantially greater than the loss for other epochs.
    merged_mae_lists = mae_training[1:] + mae_validation[1:]
    highest_loss = max(merged_mae_lists)
    lowest_loss = min(merged_mae_lists)
    delta = highest_loss - lowest_loss
    print(delta)

    top_of_y_axis = highest_loss + (delta * 0.05)
    bottom_of_y_axis = lowest_loss - (delta * 0.05)

    plt.ylim([bottom_of_y_axis, top_of_y_axis])
    plt.show()


def read_training_csv():
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
    scale_factor = 1000.0

    # Scale the training set's label.
    training_df["median_house_value"] /= scale_factor
    # Print the first rows of the pandas DataFrame.
    training_df.head(n=1000)
    training_df.describe()
    return training_df


def read_test_csv():
    pd.options.display.max_rows = 10
    pd.options.display.float_format = "{:.1f}".format
    # 获取文件目录
    curPath = os.path.abspath(os.path.dirname(__file__))
    # 获取项目根路径，内容为当前项目的名字
    rootPath = curPath[:curPath.find('machine-learning-note') + len('machine-learning-note')]
    # # 构建 CSV 文件的完整路径
    file_path = os.path.join(rootPath, 'src/resouces/california_housing_test.csv')
    # # 读取 CSV 文件
    test_df = pd.read_csv(file_path)
    scale_factor = 1000.0

    # Scale the training set's label.
    test_df["median_house_value"] /= scale_factor
    # Print the first rows of the pandas DataFrame.
    test_df.head()
    test_df.describe()
    return test_df


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
