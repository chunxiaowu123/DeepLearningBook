# 第一章 线性代数

import numpy as np

A = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 3.0]])
A_t = A.transpose()
print("A:", A)
print("A 的转置：", A_t)

m1 = np.array([[1.0, 3.0], [1.0, 0.0]])
m2 = np.array([[1.0, 2.0], [5.0, 0.0]])
print("按矩阵乘法规则：", np.dot(m1, m2))
print("按逐元素相乘：", np.multiply(m1, m2))
print("按逐元素相乘：", m1*m2)
print("按矩阵乘法规则：", m1@m2)














































