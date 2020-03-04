import numpy as np
import DNN

## 1.1
x = np.array([1,2,3])
x.__class__
x.shape
x.ndim

W = np.array([[1,2],[4,5]])
# W.shape
# W.ndim
X = np.array([[2,3],[3,4]])
print(W + X)
print(W * X)
print(np.dot(W, X)) # 行列計算
A = np.array([[1,2],[3,4]])
print(A * 10)

W1 = np.random.randn(3,4)
model = DNN()
DNN.Sigmoid.forward()


