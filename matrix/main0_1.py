from copy import deepcopy
from math import sqrt
import numpy as np
from numpy import linalg as LA
from decimal import Decimal
import numpy as np
import scipy
from scipy import linalg


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
        mtxStr += f'-------------- matrix --------------\n'
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

    def change(self, el, index, opt=''):
        if opt == 'row':
            for col in range(self.cols):
                self[col] = el[col]
        elif opt == 'col':
            for row in range(self.rows):
                self[row][index] = el[row][0]
        else:
            return 'error'

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

    def mini_det(self):
        res = 0
        flag = True
        for col in range(self.cols):
            a_1_col = self[0][col]
            tmat = []
            for trow in range(1, self.rows):
                for tcol in range(self.cols):
                    if tcol != col:
                        tmat.append(self[trow][tcol])
            if flag:
                res += a_1_col * (tmat[0] * tmat[3] - tmat[1] * tmat[2])
                flag = False
            else:
                res -= a_1_col * (tmat[0] * tmat[3] - tmat[1] * tmat[2])
                flag = True
        return res

    def determinant(self):
        res = 0
        flag = True
        for main_col in range(self.cols):
            a_1_col = self[0][main_col]
            tmat = []
            for row in range(1, self.rows):
                for col in range(self.cols):
                    if col != main_col:
                        tmat.append(self[row][col])
            tmat = Matrix([self.rows - 1, self.cols - 1], tmat)

            if tmat.rows == tmat.cols == 3:
                if flag:
                    res += a_1_col * tmat.mini_det()
                    flag = False
                else:
                    res -= a_1_col * tmat.mini_det()
                    flag = True
            else:
                tmat.determinant()
        return res

        # res = 0
        # for a in range(self.cols):
        #     temp_mat = []
        #     for row in range(self.rows):
        #         for col in range(self.cols):
        #             if row != 0 and col != a:
        #                 temp_mat.append(self[row][col])
        #     temp_mat = Matrix([len(self[a + 1::]), len(self[a + 1::])], temp_mat)
        #     print(temp_mat)
        #     flag = True
        #     if temp_mat.rows == temp_mat.cols == 3:
        #         for col in range(temp_mat.cols):
        #             a_1_col = temp_mat[0][col]
        #             tmat = []
        #             for trow in range(temp_mat.rows):
        #                 for tcol in range(temp_mat.cols):
        #                     if trow != 0 and tcol != temp_mat[0][col]:
        #                         tmat.append(temp_mat[trow][tcol])
        #             if flag:
        #                 res += a_1_col * (tmat[0] * tmat[2] - tmat[1] * tmat[3])
        #                 flag = False
        #             else:
        #                 res -= a_1_col * (tmat[0] * tmat[2] - tmat[1] * tmat[3])
        #                 flag = True
        #         print(res)
        #     temp_mat.determinant()

    def scalar_multiplication(self, b):
        res = 0
        for row in range(self.rows):
            res += self[row][0] * b[row][0]
        return res

    def projection(self, b):
        res = self.scalar_multiplication(b)
        denumirator = self.scalar_multiplication(self)
        res /= denumirator
        res = self * res
        return res

    def ortoganal_set(self):
        P = Matrix([self.rows, 1], 0)
        res = Matrix([self.rows, self.cols], 0)
        cur_b = self.get_col(0)
        res.change(cur_b, 0, 'col')
        for main_col in range(1, self.cols):
            cur_a = self.get_col(main_col)
            for col in range(main_col):
                cur_b = res.get_col(col)
                P_cur = cur_b.projection(cur_a)
                P = P + P_cur

            cur_b = cur_a - P
            res.change(cur_b, main_col, 'col')
            P = Matrix([self.rows, 1], 0)
        return res

    def normalization(self):
        res = deepcopy(self)
        for col in range(res.cols):
            cur_col = res.get_col(col=col)
            cur_col_length = 0
            for rw in range(cur_col.rows):
                cur_col_length += cur_col[rw][0] ** 2
            cur_col_length = 1 / sqrt(cur_col_length)
            res = res.mul_col_by_el(col, cur_col_length)
        return res

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
            try:
                if res[i][i] == 0:
                    res = res.swap_max_el_matrix(i)
                    if res[i][i] == 0:
                        n += 1
            except IndexError:
                res = res.del_zero_rows()
                return res
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
                # b_col = A_col * b_col.transposition()ё
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

    def QR_decomposition(self):
        A = deepcopy(self)
        ort_set = A.ortoganal_set()
        Q = ort_set.normalization()
        Q_T = Q.transposition()
        R = Q_T * A
        return Q, R



# exGauss = Matrix([5, 4], [2, 1, -2, 6, 3, 0, 0, -1, 1, -1, 2, -7, 5, -2, 4, -15, 7, 2, -4, 11])
exGauss = Matrix([4, 3], [4, 4, 8, 2, 2, 4, 3, 1, 0, 3, 7, 6])
exGauss = Matrix([3, 3], [1, 2, 3, 3, 1, 5, 2, 4, 6])
exGauss = Matrix([3, 6], [0, 2, 0, 3, 4, 1, 2, 4, 4, 6, 8, 2, 3, 6, 6, 9, 12, 3])
# exGrevil = Matrix([4, 3], [1, -1, 0, -1, 2, 1, 2, -3, -1, 0, 1, 1])
# exGrevil = Matrix([3, 4], [2, 1, 1, 3, 1, 0, 1, -1, 1, 1, 0, 4])
# exGrevil = Matrix([4, 3], [1, -1, 0, -1, 2, 1, 2, -3, -1, 0, 1, 1])
# exGrevil = Matrix([3, 4], [1, -1, 2, 0, -1, 2, -3, 1, 0, 1, -1, 1])
# exGauss = Matrix([3, 4], [1, 2, 3, 4, 1, 2, 5, 6, 3, 6, 13, 16])
# ex_skeletal = Matrix([4, 4], [3, -3, -5, 8, -3, 2, 4, -6, 2, -5, -7, 5, -4, 3, 5, -6])
# ex_skeletal = Matrix([3, 3], [1, -2, 3, 4, 0, 6, -7, 8, 9])
# ex_skeletal = Matrix([4, 3], [1, 1, 0, -1, 1, -1, 0, 1, 1, 1, 1, 1])
# ex_skeletal = Matrix([3, 3], [12, -51, 4, 6, 167, -68, -4, 24, -41])
# ex_skeletal = Matrix([2, 2], [1, 2, 3, 4])
print('__________________________Q______________________________')
a = np.array([[2, 3, 5], [7, 11, 13], [17, 19, 23]])
q, r = np.linalg.qr(a)
print(q, '\n-----------------------R------------------------\n', r)
ex_1 = Matrix([3, 3], [2, 3, 5, 7, 11, 13, 17, 19, 23])
Q, R = ex_1.QR_decomposition()
print(Q, '\n', R)
