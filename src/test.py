l = [[1, 2, 3], [4, 5, 0], [7, 8, 12], [10, 11 ,-1]]
l = list(enumerate(l))
print(l)
print(max(l, key=lambda x: x[1][2]))
