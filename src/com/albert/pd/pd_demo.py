import numpy as np
import pandas as pd


def pd_demo_fun():
    my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])
    # Create a Python list that holds the names of the two columns.
    my_column_names = ['温度', '活动']

    # Create a DataFrame.
    my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

    # Print the entire DataFrame
    print(my_dataframe)

    my_dataframe['调整温度'] = my_dataframe['活动'] + 2
    print(my_dataframe)

    print("Rows #0, #1, and #2:")
    print(my_dataframe.head(3), '\n')

    print("Row #2:")
    print(my_dataframe.iloc[[2]], '\n')

    print("Rows #1, #2, and #3:")
    print(my_dataframe[1:4], '\n')

    print(my_dataframe['温度'])


def pd_demo_task_1():
    # 1. 创建一个 3x4（3 行 x 4 列）的 pandas DataFrame，其中的列名为`Eleanor`、
    #     `Chidi`、`Tahani`和`Jason`。使用 0 到 100 之间的随机整数（包括 0 和 100）填充 DataFrame 中的 12 个单元格中的每一个。
    # 2. 输出以下内容：
    #     - 整个 DataFrame
    #     - `Eleanor` 该列第 1 行单元格中的值
    # 3. 创建第五列，名为，其中填充了和`Janet`的逐行总和。`TahaniJason`
    # 4. 为了完成此任务，了解 NumPy UltraQuick 教程中涵盖的 NumPy 基础知识会有所帮助。
    my_data = np.random.randint(low=0, high=100, size=(3, 4))
    my_column_names = ['Eleanor', 'Chidi', 'Tahani', 'Jason']
    my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)
    my_dataframe['TahaniJason'] = my_dataframe['Tahani'] + my_dataframe['Jason']
    print(my_dataframe)
    my_dataframe_of_copy = my_dataframe.copy()
    print(my_dataframe_of_copy)
class PdDemo:
    pass
