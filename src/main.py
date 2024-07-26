# from src.com.albert.pd import pd_demo
# from src.com.albert.np import np_demo
# from src.com.albert.colab import colab_demo
from  src.com.albert.lopd import  law_of_probability_distribution


if __name__ == "__main__":
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
    # my_model = colab_demo.build_model(my_learning_rate)
    # trained_weight, trained_bias, epochs, root_mse = colab_demo.train_model(my_model, my_feature,my_label, epochs,my_batch_size)
    # colab_demo.plot_the_model(trained_weight, trained_bias, my_feature, my_label)
    # colab_demo.plot_the_loss_curve(epochs, root_mse)
    # =========================================my_batch_size ====================================
    # my_learning_rate = 0.1
    # epochs = 30
    # my_batch_size = 12
    # my_feature = "median_income"
    # my_label = "median_house_value"
    # my_model = None
    # training_df = colab_demo_task2.read_csv()
    # my_model = colab_demo_task2.build_model(my_learning_rate)
    # trained_weight, trained_bias, epochs, root_mse = colab_demo_task2.train_model(my_model, training_df, my_feature,
    #                                                                               my_label, epochs, my_batch_size)
    # colab_demo_task2.plot_the_model(trained_weight, training_df, trained_bias, my_feature, my_label)
    # colab_demo_task2.plot_the_loss_curve(epochs, root_mse)
    # =========================================== 概率分布 =====================
    law_of_probability_distribution.lopd_demo()
