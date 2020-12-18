import numpy as np


def sigmoid(x):
    """Сигмоидная функция активации (логичтическая)"""
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weight, bias):
        """Конструктор класса нейрон"""
        self.weight = weight
        self.bias = bias

    def feed_forward(self, inputs):
        """Конструктор класса нейрон"""
        res = np.dot(self.weight, inputs) + self.bias
        return sigmoid(res)


weights = np.array([0, 1])
bias = 4
n = Neuron(weights, bias)

x = np.array([2, 3])
print(n.feed_forward(x))
