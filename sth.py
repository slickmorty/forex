import tensorflow
import numpy as np

a = [0, 1, 0, 1, 0]


a = np.asarray(a)
depth = 2


b = tensorflow.one_hot(a, depth)


print(b)
