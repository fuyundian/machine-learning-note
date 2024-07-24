import numpy as np

def np_demo():
    two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])
    print(two_dimensional_array)
    sequence_of_integers = np.arange(5, 12)
    print(sequence_of_integers)
    random_integers_between_50_and_100 = np.random.randint(low=50, high=101, size=(6,))
    print(random_integers_between_50_and_100)
    random_floats_between_0_and_1 = np.random.random((6,))
    print(random_floats_between_0_and_1)
    random_floats_between_2_and_3 = random_floats_between_0_and_1 + 2.0
    print(random_floats_between_2_and_3)
    random_integers_between_150_and_300 = random_integers_between_50_and_100 * 3
    print(random_integers_between_150_and_300)

class NpDemo :
    pass