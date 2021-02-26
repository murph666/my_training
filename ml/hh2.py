import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


# =============================================================================
# Реализовать функцию, осуществляющую матричные операции для получения theta.
# посчитать линейную регрессию
# =============================================================================


def linreg_linear(X, y):
    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

# =============================================================================
# У какого из признаков наибольшее стандартное отклонение, чему оно равно
# =============================================================================


def max_feature(data):
    table, feature = data['data'], data['feature_names']
    index_of_max_value = np.where(
        table.std(axis=0) == np.amax(table.std(axis=0)))
    return feature[int(index_of_max_value[0][0])], np.amax(X.std(axis=0))


def print_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f'MSE = {mse:.2f}, RMSE = {rmse:.2f}')


data = load_boston()

X, y = data['data'], data['target']

#X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)


theta = linreg_linear(x_train, y_train)

y_pred = x_valid.dot(theta)
y_train_pred = x_train.dot(theta)

print('----my_reg----')
print_regression_metrics(y_valid, y_pred)
print_regression_metrics(y_train, y_train_pred)

lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)
print('----sklern----')
print_regression_metrics(y, y_pred)


index_of_feature = np.where(data['feature_names'] == 'B')
sorted_X = X[:, index_of_feature[0][0]]
print(sorted_X[sorted_X > 50])

