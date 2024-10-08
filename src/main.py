# from src.com.albert.pd import pd_demo
# from src.com.albert.np import np_demo
# from src.com.albert.colab import build_model
# from src.com.albert.colab import test_sets
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from src.com.albert.lopd import law_of_probability_distribution
from src.com.albert.colab import validation_and_test_sets

if __name__ == "__main__":
    print("=============================================== start ===============================================")
    # numpy 使用
    # np_demo.np_demo()
    # pd_demo.pd_demo_fun()
    # pd 任务1
    # pd_demo.pd_demo_task_1()
    # my_learning_rate = 0.14
    # epochs = 70
    # my_batch_size = 12
    #
    # my_feature = np.linspace(0, 10, 10)  # 生成 10 个在 0 到 10 之间均匀分布的点
    # my_label = 2 * my_feature + 1 + np.random.normal(0, 1, size=10)
    # # my_batch_size = 12
    # # my_learning_rate = 0.01
    # # epochs = 10
    # # my_feature = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    # # my_label = np.array([5.0, 8.8, 9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])
    # my_model = build_model.build_model(my_learning_rate)
    # trained_weight, trained_bias, epochs, root_mse = build_model.train_model(my_model, my_feature,my_label, epochs,my_batch_size)
    # build_model.plot_the_model(trained_weight, trained_bias, my_feature, my_label)
    # build_model.plot_the_loss_curve(epochs, root_mse)
    # =========================================my_batch_size ====================================
    # my_learning_rate = 0.1
    # epochs = 30
    # my_batch_size = 12
    # my_feature = "median_income"
    # my_label = "median_house_value"
    # my_model = None
    # training_df = colab_demo_task2.read_csv()
    # my_model = test_sets.build_model(my_learning_rate)
    # trained_weight, trained_bias, epochs, root_mse = test_sets.train_model(my_model, training_df, my_feature,
    #                                                                               my_label, epochs, my_batch_size)
    # test_sets.plot_the_model(trained_weight, training_df, trained_bias, my_feature, my_label)
    # test_sets.plot_the_loss_curve(epochs, root_mse)
    # =========================================== 概率分布 =====================
    # law_of_probability_distribution.lopd_demo()
    # =========================================== 验证和测试集.ipynb =====================
    my_learning_rate = 0.08
    my_epochs = 30
    my_batch_size = 100

    validation_split = 0.2

    my_feature = "median_income"
    my_label = "median_house_value"
    training_df = validation_and_test_sets.read_training_csv()
    test_df = validation_and_test_sets.read_test_csv()
    my_model = validation_and_test_sets.build_model(my_learning_rate)
    epochs, root_mse, history = validation_and_test_sets.train_model(my_model, training_df, my_feature, my_label,
                                                                     my_epochs, my_batch_size, validation_split)
    validation_and_test_sets.plot_the_loss_curve(epochs, history["root_mean_squared_error"],
                                                 history["val_root_mean_squared_error"])
    learning_rate = 0.08
    epochs = 70
    batch_size = 100

    validation_split = 0.2

    my_feature = "median_income"
    my_label = "median_house_value"
    shuffled_train_df = training_df.reindex(np.random.permutation(training_df.index))

    my_model = validation_and_test_sets.build_model(learning_rate)
    epochs, rmse, history = validation_and_test_sets.train_model(my_model, shuffled_train_df, my_feature, my_label,
                                                                 epochs, batch_size, validation_split)

    validation_and_test_sets.plot_the_loss_curve(epochs, history["root_mean_squared_error"],
                                                 history["val_root_mean_squared_error"])
    print("=============================================== end =================================================")
