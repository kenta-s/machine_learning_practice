import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

rng = np.random.RandomState(123)

d = 2
N = 10
mean = 5

x1 = rng.randn(N, d) + np.array([0, 0])
x2 = rng.randn(N, d) + np.array([mean, mean])

x = np.concatenate((x1, x2), axis=0)

plt.plot(x1.T[0], x1.T[1], 'ro', x2.T[0], x2.T[1], 'b^')
plt.axis([-4, 8, -6, 12])
plt.show()
