import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def linreg_linear(X, y):
    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))


data = load_boston()
print(data.keys())
X, y = data['data'], data['target']

X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])
theta = linreg_linear(X, y)
print(theta)


# %%
# print(np.amax(X.std(axis=1)))
print('------------------------0-----------------------', X.std(axis=0), sep='\n')
print('------------------------1-----------------------', X.std(axis=1), sep='\n')
