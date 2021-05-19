# 第一章 线性代数

import numpy as np

A = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 3.0]])
A_t = A.transpose()
print("A:", A)
print("A 的转置：", A_t)

# 矩阵乘法
m1 = np.array([[1.0, 3.0], [1.0, 0.0]])
m2 = np.array([[1.0, 2.0], [5.0, 0.0]])
print("按矩阵乘法规则：", np.dot(m1, m2))
print("按逐元素相乘：", np.multiply(m1, m2))
print("按逐元素相乘：", m1*m2)
print("按矩阵乘法规则：", m1@m2)


# 单位矩阵
print(np.identity(3))

# 矩阵的逆
A = [[1.0, 2.0], [3.0, 4.0]]
A_inv = np.linalg.inv(A)
print("A的逆矩阵", A_inv)

a = np.array([1.0, 3.0])
print("向量2范数", np.linalg.norm(a, ord=2))
print("向量1范数", np.linalg.norm(a, ord=1))
print("向量无穷范数", np.linalg.norm(a, ord=np.inf))












































