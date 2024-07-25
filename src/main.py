# from src.com.albert.pd import pd_demo
# from src.com.albert.np import np_demo
import numpy as np

from src.com.albert.colab import colab_demo

if __name__ == "__main__":
    # numpy 使用
    # np_demo.np_demo()
    # pd_demo.pd_demo_fun()
    # pd 任务1
    # pd_demo.pd_demo_task_1()
    my_learning_rate = 0.01
    epochs = 100
    my_batch_size = 12

    my_feature = np.linspace(0, 10, 100)  # 生成 100 个在 0 到 10 之间均匀分布的点
    my_label = 2 * my_feature + 1 + np.random.normal(0, 1, size=100)
    # my_batch_size = 12
    # my_learning_rate = 0.01
    # epochs = 10
    # my_feature = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    # my_label = np.array([5.0, 8.8, 9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])
    my_model = colab_demo.build_model(my_learning_rate)
    trained_weight, trained_bias, epochs, rmse = colab_demo.train_model(my_model, my_feature,
                                                                        my_label, epochs,
                                                                        my_batch_size)
    colab_demo.plot_the_model(trained_weight, trained_bias, my_feature, my_label)
    colab_demo.plot_the_loss_curve(epochs, rmse)
    # x_train = np.linspace(0, 10, 100)  # 生成 100 个在 0 到 10 之间均匀分布的点
    # y_train = 2 * x_train + 1 + np.random.normal(0, 1, size=100)  # 生成对应的 y 值，并添加一些噪声
    #
    # # 构建神经网络模型
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(1, input_shape=(1,))  # 一个神经元的全连接层，输入维度为 1
    # ])
    #
    # # 编译模型
    # model.compile(optimizer='RMSprop', loss='mse')  # 使用均方误差作为损失函数，adam 优化器
    #
    # # 训练模型
    # history = model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=0)  # 训练 50 个 epoch，批量大小为 10
    #
    # # 预测并绘制拟合曲线
    # x_pred = np.linspace(0, 10, 200)  # 生成用于预测的 x 值
    # y_pred = model.predict(x_pred)  # 进行预测
    #
    # # 绘制训练过程中的损失变化
    # plt.plot(history.history['loss'])
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.show()
    #
    # # 绘制原始数据和拟合曲线
    # plt.scatter(x_train, y_train, label='Original Data')
    # plt.plot(x_pred, y_pred, color='red', label='Fitted Line')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Linear Regression')
    # plt.legend()
    # plt.show()
