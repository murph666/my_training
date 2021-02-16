from functools import reduce


def almost_double_factorial(n):
    return reduce(lambda s, x: s * x, list(filter(lambda x: x % 2 != 0, [i for i in range(n)] if n != 0 else [1])))


print(almost_double_factorial(0))
