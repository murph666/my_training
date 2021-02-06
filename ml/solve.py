import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

# data = np.array([100, 100, 110, 150, 130])
# data = data.reshape(-1, 1)
# print(data)
# print(MinMaxScaler().fit_transform(data))

data = {'Страна': ['russia', 'Ukraine', 'russia', 'Ukraine', 'russia'],
        'Дата': ['date', 'date', 'date', 'date', 'date'],
        'Кол-во звезд': [5, 5, 4, 4, 5],
        'Кол-во людей': [1, 2, 3, 4, 5],
        'Стоймость': [120000, 241526, 80698, 140088, 228322]}
frame = pd.DataFrame(data)
print('-------------------Исходный датасет--------------------', frame, sep='\n')
print('-------------------------------------------------------')

# Применим команды обработки данных. Сделал копию что бы не юзать исходник. Искусствено
# добавил пропуск
data1 = deepcopy(data)
data1['Стоймость'][2] = np.nan
frame = pd.DataFrame(data1)
print(frame, frame.fillna(np.mean(frame['Стоймость'])).sort_values(by='Стоймость'),
      sep='\n ----------------уберем Nan и отсортируем--------------\n')
frame = frame.fillna(np.mean(frame['Стоймость']))

# При нормальзации необходимо подавать 2D массив. Поэтому надо решейпить столбец
# стоймости. Решейп только к нп.аррай юзается
# MinMaxScaler и StandardScaler возвращает(не изменяет исходый!) измененный массив
print('-------------------Примение нормализации--------------------')
print('\t\t\t\t\tMinMaxScaler')
frame['Стоймость'] = MinMaxScaler().fit_transform(np.array(frame['Стоймость']).reshape(-1, 1))
print(frame)
frame.hist()
print('------------------------------------------------------------')
print('\t\t\t\t\tStandardScaler')
frame['Стоймость'] = StandardScaler().fit_transform(np.array(frame['Стоймость']).reshape(-1, 1))
print(frame)
print('------------------------------------------------------------')

