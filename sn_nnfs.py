import numpy as np
import nnfs
nnfs.init()
from nnfs.datasets import spiral_data

import matplotlib.pyplot as plt

X,y = spiral_data(samples=100, classes=4)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()
