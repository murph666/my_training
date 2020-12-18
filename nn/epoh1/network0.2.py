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
        epochs = 1000
        y_pred = 0

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                out_h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
                out_h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b1)
                out = sigmoid(self.w5 * out_h1 + self.w6 * out_h2 + self.b1)
                y_pred = out

                d_L_d_ypred = -2 * (y_true - y_pred)

                d_ypred_d_w5 = out_h1 * derivative_of_sigmoid(out)
                d_ypred_d_w6 = out_h2 * derivative_of_sigmoid(out)
                d_ypred_d_b3 = derivative_of_sigmoid(out)

                d_ypred_d_h1 = self.w5 * derivative_of_sigmoid(out)
                d_ypred_d_h2 = self.w6 * derivative_of_sigmoid(out)

                d_h1_d_w1 = x[0] * derivative_of_sigmoid(out_h1)
                d_h1_d_w2 = x[1] * derivative_of_sigmoid(out_h1)
                d_h1_d_b1 = derivative_of_sigmoid(out_h1)

                d_h2_d_w3 = x[0] * derivative_of_sigmoid(out_h2)
                d_h2_d_w4 = x[1] * derivative_of_sigmoid(out_h2)
                d_h2_d_b2 = derivative_of_sigmoid(out_h2)

                # --- Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feed_forward, 1, data)
                    loss = mse_loss(all_y_trues, y_preds)
                    print("Epoch %d loss: %.3f" % (epoch, loss))


data = np.array([[-2, -1], [25, 6], [17, 4], [-15, -6]])
all_y_trues = np.array([1, 0, 0, 1])

network = NeuralNetwork()
network.train(data, all_y_trues)
emily = np.array([-7, -3])
print("Emily: %.3f" % network.feed_forward(emily))
