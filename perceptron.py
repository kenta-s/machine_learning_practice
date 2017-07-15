import numpy as np

rng = np.random.RandomState(123)

d = 2
N = 10
mean = 5

x1 = rng.randn(N, d) + np.array([0, 0])
x2 = rng.randn(N, d) + np.array([mean, mean])

x = np.concatenate((x1, x2), axis=0)

# import matplotlib.pyplot as plt
# 
# plt.plot(x1, x2, 'o')
# plt.show()
