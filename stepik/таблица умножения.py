# a, b, c, d = int(input()), int(input()), int(input()), int(input())
# string = str()
# for j in range(a - 1, b + 1):
#     for i in range(c - 1, d + 1):
#         if j == a - 1 and i == c - 1:
#             string += '\t'
#         elif j == a - 1:
#             string += f'{i}\t'
#         elif i == c - 1:
#             string += f'{j}\t'
#         else:
#             string += f'{i * j}\t'
#     print(string)
#     string = ''

a, b, c, d = (int(input()) for x in range(4))
pass
print('', *range(c, d + 1), sep='\t')
for x in range(a, b + 1):
    print(x, *[y * x for y in range(c, d + 1)], sep='\t')
