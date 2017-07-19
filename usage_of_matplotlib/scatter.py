import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

x = np.random.rand(100)
y = np.random.rand(100)

plt.scatter(x, y)
# change the marker
# plt.scatter(x, y, marker='*')

# change color
# plt.scatter(x, y, c='pink')

# change size
# plt.scatter(x, y, s=600)

# Add title
plt.title("This is a title")

# Add X label
plt.xlabel("x axis")

# Add Y label
plt.ylabel("y axis")

# Add grid
plt.grid(True)
plt.show()
