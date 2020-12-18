import numpy as np


def sigmoid(x):
    """Сигмоидная функция активации (логичтическая)"""
    return 1 / (1 + np.exp(-x))


def derivative_of_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(true, predict):
    return ((true - predict) ** 2).mean()  # .mean возьмет ср.ар


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
        # Инициализируем веса
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        # Инициализируем пороги
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feed_forward(self, x):
        out_h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        out_h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b1)
        out = sigmoid(self.w5 * out_h1 + self.w6 * out_h2 + self.b1)
        return out

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 100

        for epoh in range(epochs):
            pass

