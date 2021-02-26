import numpy as np
from matplotlib import pyplot as plt

a = 3
numiter = 40


def y(x, a): return a*x*(1-x)
def g(x): return x


x = np.linspace(0, 1, 100)
y_res = np.array([])
for n in range(1, numiter):
    np.append(y_res, y(n, a))


fig, ax = plt.subplots()
ax.plot(x, g(x), x, y(x, a))
ax.set_title('Simple plot')
plt.show()