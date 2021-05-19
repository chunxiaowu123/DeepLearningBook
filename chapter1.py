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


# 特征值分解
A = np.array([[1.0,2.0,3.0],
              [4.0,5.0,6.0],
              [7.0,8.0,9.0]])

# 计算特征值
print("特征值:", np.linalg.eigvals(A))
# 计算特征值和特征向量
eigvals,eigvectors = np.linalg.eig(A)
print("特征值:", eigvals)
print("特征向量:", eigvectors)

# 奇异值分解
A = np.array([[1.0,2.0],
              [4.0,5.0],
              [3, 6]])
U,D,V = np.linalg.svd(A)
print("U:", U)
print("D:", D)
print("V:", V)

dd = np.zeros((3,2))
dd[0,0] = D[0]
dd[1,1] = D[1]
print(U@dd@V)

# PCA主成分分析
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 载入数据
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
df.label.value_counts()

# 查看数据
df.tail()

# 查看数据
X = df.iloc[:, 0:4]
y = df.iloc[:, 4]
print("查看第一个数据： \n", X.iloc[0, 0:4])
print("查看第一个标签: \n", y.iloc[0])

class PCA():
    def __init__(self):
        pass

    def fit(self, X, n_components):
        n_samples = np.shape(X)[0]
        covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(X - X.mean(axis=0))
        # 对协方差矩阵进行特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # 对特征值（特征向量）从大到小排序
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]
        # 得到低维表示
        X_transformed = X.dot(eigenvectors)
        return X_transformed

model = PCA()
Y = model.fit(X, 2)

principalDf = pd.DataFrame(np.array(Y),
                           columns=['principal component 1', 'principal component 2'])
Df = pd.concat([principalDf, y], axis = 1)
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1, 2]
# ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = Df['label'] == target
    ax.scatter(Df.loc[indicesToKeep, 'principal component 1']
                , Df.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50)
ax.legend(targets)
ax.grid()



from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y = sklearn_pca.fit_transform(X)

principalDf = pd.DataFrame(data = np.array(Y), columns = ['principal component 1', 'principal component 2'])
Df = pd.concat([principalDf, y], axis = 1)
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1, 2]
# ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = Df['label'] == target
    ax.scatter(Df.loc[indicesToKeep, 'principal component 1']
                , Df.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50)
ax.legend(targets)
ax.grid()
plt.show()







































