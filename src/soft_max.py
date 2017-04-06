"""Softmax."""
import numpy as np
#scores = [3.0, 1.0, 0.2]
scores = np.array([[1,2,3,6],[2,4,5,7],[3,9,3,6]])


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x/10, np.ones_like(x)/10, 0.2 * np.ones_like(x)/10])
print softmax(scores).T
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
