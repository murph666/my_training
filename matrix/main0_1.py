from copy import deepcopy
from decimal import Decimal
import numpy as np


class Matrix:
    def __init__(self, shape, fill):
        self.rows = shape[0]
        self.cols = shape[1]
        self.fill = fill
        if isinstance(self.fill, (int, float)):
            self.matrix = [[self.fill for j in range(self.cols)] for i in range(self.rows)]
        elif isinstance(self.fill[0], (int, float)):
            el = iter(self.fill)
            self.matrix = [[next(el) for j in range(self.cols)] for i in range(self.rows)]
        elif (isinstance(self.fill, tuple)):
            temp_list = []
            for mas in range(len(self.fill)):
                el = self.fill[mas]
                for row in range(len(el)):
                    for col in range(len(el[0])):
                        temp_list.append(el[row][col])
            self.fill = temp_list
            el = iter(self.fill)
            self.matrix = [[next(el) for j in range(self.cols)] for i in range(self.rows)]
        elif (isinstance(self.fill[0], list)):
            temp_list = []
            for row in range(len(self.fill)):
                el = self.fill[row]
                for col in range(len(el)):
                    temp_list.append(el[col])
            self.fill = temp_list
            el = iter(self.fill)
            self.matrix = [[next(el) for j in range(self.cols)] for i in range(self.rows)]
        else:
            return 'error_matrix_value'

    def __str__(self):
        mtxStr = ''
        mtxStr += '-------------- matrix --------------\n'
        for i in range(len(self.matrix)):
            mtxStr += (' '.join(map(lambda x: '{0:8.3f}'.format(x), self.matrix[i])) + '\n')
        mtxStr += '------------------------------------'
        mtxStr += '\n'
        return mtxStr

    def __getitem__(self, index):
        return self.matrix[index]

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            i = index[0]
            j = index[1]
            self.matrix[i][j] = value
        elif isinstance(index, int):
            self.matrix[index] = value

    def __add__(self, other):
        res = Matrix((self.rows, self.cols), 0)
        if isinstance(other, Matrix):
            for i in range(self.rows):
                for j in range(self.cols):
                    res.matrix[i][j] = self.matrix[i][j] + other.matrix[i][j]
        elif isinstance(other, (int, float)):
            for i in range(self.rows):
                for j in range(self.cols):
                    res.matrix[i][j] = self.matrix[i][j] + other

        return res

    def __sub__(self, other):
        res = Matrix((self.rows, self.cols), 0)
        if isinstance(other, Matrix):
            for i in range(self.rows):
                for j in range(self.cols):
                    res.matrix[i][j] = self.matrix[i][j] - other.matrix[i][j]
        elif isinstance(other, (int, float)):
            for i in range(self.rows):
                for j in range(self.cols):
                    res.matrix[i][j] = self.matrix[i][j] - other

        return res

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.cols == other.rows:
                temp_rows = self.rows
                temp_cols = other.cols
                res = Matrix((temp_rows, temp_cols), 0)
                for row in range(res.rows):
                    for col in range(res.cols):
                        temp_el = 0
                        for j in range(self.cols):
                            temp_el += self.matrix[row][j] * other.matrix[j][col]
                        res[row][col] = temp_el
                return res
            elif (other.rows == 1) and (other.cols == 1):
                res = Matrix((self.rows, self.cols), 0)
                for row in range(self.rows):
                    for col in range(self.cols):
                        res[row][col] = self.matrix[row][col] * other[0][0]
                        if abs(res[row][col]) < 0.001: res[row][col] = 0
                return res

        elif isinstance(other, (int, float)):
            res = Matrix((self.rows, self.cols), 0)
            for row in range(self.rows):
                for col in range(self.cols):
                    res[row][col] = self.matrix[row][col] * other
            return res

            # elif self.rows == other.cols:
            #     temp_rows = other.rows
            #     temp_cols = self.cols
            #     res = Matrix((temp_rows, temp_cols), 0)
            #
            #     for col_self in range(self.cols):
            #         for row_self in range(self.rows):
            #             el = self[row_self][col_self]
            #             temp_el = 0
            #             temp_list = []
            #             for row_other in range(other.cols):
            #                 for col_other in range(other.rows):
            #                     temp_el += el * other[row_other][col_other]
            #             temp_list.append(temp_el)
            #     res = Matrix((other.rows, other.cols), temp_list)

    def swap_max_el_matrix(self, count):
        max_el = abs(self.matrix[count][count])
        max_row = count
        res = deepcopy(self)
        for i in range(count, len(res.matrix)):
            if abs(res.matrix[i][count]) > max_el:
                max_el = abs(res.matrix[i][count])
                max_row = i
        res.matrix[count], res.matrix[max_row] = res.matrix[max_row], res.matrix[count]
        return res

    def mul_row_by_el(self, row, el):
        res = deepcopy(self)
        if isinstance(el, (int, float)):
            for i in range(len(res[row])):
                res[row][i] = res[row][i] * el
                if abs(res[row][i]) < 0.001:
                    res[row][i] = 0
        elif isinstance(el[0], list):
            if self.cols == len(el):
                for i in range(len(res[row])):
                    res[row][i] = res[row][i] * el[i][0]
            else:
                return 'error: n not equal m'
        return res

    def mul_col_by_el(self, col, el):
        res = deepcopy(self)
        if isinstance(el, (int, float)):
            for i in range(res.rows):
                res[i][col] = res[i][col] * el
                if abs(res[i][col]) < 0.0001:
                    res[i][col] = 0
        elif isinstance(el, list):
            for i in range(self.rows):
                res[i][col] = res[i][col] * el[i]
        else:
            return 'error: m not equal n'
        return res

    def del_zero_rows(self):
        temp_mas = []
        for i in range(self.rows):
            flag_zero = False
            for j in range(self.cols):
                if abs(self[i][j]) > 0.001:
                    flag_zero = True
                    break
            if flag_zero:
                temp_mas.append(self[i])
        res = Matrix((len(temp_mas), len(temp_mas[0])), temp_mas)
        return res

    def del_zero_cols(self):
        temp_mas = [[] for row in range(self.rows)]
        for i in range(self.cols):
            temp_list = []
            count = 0
            for j in range(self.rows):  # Выделили столбец
                temp_list.append(self[j][i])
            for k in range(len(temp_list)):  # Увеличиваем count если элемент = 0
                if temp_list[k] == 0:
                    count += 1
            if count != self.rows:  # Если count != количеству строк, т.е не все элементы выделенного столбца != 0
                for k in range(len(temp_list)):
                    temp_mas[k].append(temp_list[k])
        res = Matrix((len(temp_mas), len(temp_mas[0])), temp_mas)
        return res

    def transposition(self, col='', row=''):
        if isinstance(col, int):
            '''
            Транспонируем выбраный col
            '''
            temp_col = []
            for i in range(self.rows):
                temp_col.append(self[i][col])
            res = Matrix((1, self.rows), temp_col)
            return res
        elif isinstance(row, int):
            '''
            Транспонируем выбранную row
            '''
            temp_row = []
            for i in range(self.cols):
                temp_row.append(self[row][i])
            res = Matrix((self.cols, 1), temp_row)
            return res
        elif isinstance(col, int) and isinstance(row, int):
            return '<error>'
        else:
            transpos_matrix = []
            for j in range(len(self.matrix[0])):
                temp = []
                for i in range(len(self.matrix)):
                    temp.append(self.matrix[i][j])
                transpos_matrix.append(temp)
            res = Matrix((self.cols, self.rows), transpos_matrix)

        return res

    def get_col(self, col='', cols=''):
        if isinstance(col, int):
            temp_el = []
            for row in range(self.rows):
                temp_el.append(self[row][col])
            res = Matrix((len(temp_el), 1), temp_el)
            return res
        elif isinstance(cols, int):
            temp_el = []
            for row in range(self.rows):
                for col in range(cols):
                    temp_el.append(self[row][col])
            res = Matrix((self.rows, cols), temp_el)
            return res

    def get_row(self, row=''):
        temp_el = []
        for r in range(row):
            temp_el.extend(self[r])
        res = Matrix((row, self.cols), temp_el)
        return res

    def invertible_matrix(self):
        for row in range(self.rows):
            for col in range(self.cols):
                self[row][row] = 1 / self[row][row]

    def check_zero(self):
        if isinstance(self, Matrix):
            if self.rows < self.cols:  # Пришла строка
                for col in range(self.cols):
                    if self[0][col] != 0:
                        return True
            elif self.rows > self.cols:  # Пришел столбец
                for row in range(self.rows):
                    if abs(self[row][0]) > 0.001:
                        if self[row][0] != 0:
                            return True
            else:
                return '<error>'

    def gauss(self):
        res = deepcopy(self)
        n = 0
        for i in range(res.rows):
            if res[i][i] == 0:
                res.swap_max_el_matrix(i)
                if res[i][i] == 0:
                    n += 1
            try:
                c = 1 / res[i][i + n]
            except ZeroDivisionError:
                res = res.del_zero_rows()
                return res
            except IndexError:
                res = res.del_zero_rows()
                return res
            res = res.mul_row_by_el(i, c)
            for j in range(i + 1, res.rows):
                an = res[j][i + n]
                for k in range(i + n, res.cols):
                    a = res[i][k] * an * -1
                    res[j][k] += a
        return res

    def grevil(self):
        res = deepcopy(self)
        a1 = self.transposition(col=0) * self.get_col(0)  # Вычисляем А1+
        a1 = 1 / a1[0][0]
        A_col = self.transposition(col=0).mul_row_by_el(0, a1)
        for col in range(1, self.cols):
            d_col = A_col * self.get_col(col)
            c_col = self.get_col(cols=col)
            c_col = c_col * d_col
            c_col = self.get_col(col) - c_col
            if c_col.check_zero():  # Проверяем с
                b_col = c_col.transposition(col=0) * c_col.get_col(0)
                b_col = 1 / b_col[0][0]
                b_col = c_col.transposition(col=0).mul_row_by_el(0, b_col)
            else:
                b_col1 = d_col.transposition() * d_col
                b_col1 = 1 / (b_col1[0][0] + 1)
                b_col = d_col.transposition()
                b_col = b_col * b_col1
                b_col = b_col * A_col
                # b_col = A_col * b_col.transposition()
                # b_col = b_col.mul_row_by_el(0, d_col1)

            B_col = d_col * b_col
            B_col = A_col - B_col
            A_col = Matrix((col + 1, B_col.cols), (B_col.matrix, b_col.matrix))
        res = deepcopy(A_col)
        return res

    def skelet_decomposition(self):
        len = self.gauss().rows
        C = self.get_row(len)
        C_speudo = C.grevil()
        res = self * C_speudo
        return res


# exGauss = Matrix([5, 4], [2, 1, -2, 6, 3, 0, 0, -1, 1, -1, 2, -7, 5, -2, 4, -15, 7, 2, -4, 11])
# exGrevil = Matrix([4, 3], [1, -1, 0, -1, 2, 1, 2, -3, -1, 0, 1, 1])
# exGrevil = Matrix([3, 4], [2, 1, 1, 3, 1, 0, 1, -1, 1, 1, 0, 4])
# exGrevil = Matrix([4, 3], [1,-1,0,-1,2,1,2,-3,-1,0,1,1])
# exGrevil = Matrix([3, 4], [1, -1, 2, 0, -1, 2, -3, 1, 0, 1, -1, 1])
# exGauss = Matrix([3, 4], [1, 2, 3, 4, 1, 2, 5, 6, 3, 6, 13, 16])
ex_skeletal = Matrix([3, 4], [2, 1, 1, 3, 1, 0, 1, -1, 1, 1, 0, 4])

# print(exGauss)
print(ex_skeletal)

print(ex_skeletal.skelet_decomposition())
# print(exGauss.gauss())


# numMatrixA = np.array([[1, 2, 3], [4, 5, 6]])
# numMatrixB = np.array([[7, 8], [9, 1], [2, 3]])
#
# print(np.dot(numMatrixA, numMatrixB))
# print(np.dot(numMatrixB, numMatrixA))
