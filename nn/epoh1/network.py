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
        """Функция прямой связи"""
        res = np.dot(self.weight, inputs) + self.bias
        return sigmoid(res)


class NeuralNetwork:
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.a = Neuron(weights, bias)

    def feed_forward(self, x):
        self.out_h1 = self.h1.feed_forward(x)
        self.out_h2 = self.h2.feed_forward(x)
        self.out = self.a.feed_forward([self.out_h1, self.out_h2])
        return self.out


n = NeuralNetwork()
x = np.array([2, 3])
print(n.feed_forward(x))
