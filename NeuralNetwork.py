import matplotlib as matplotlib
import numpy as np  # Python中科学计算的主要工具
import h5py
import matplotlib.pyplot as plt  # Python作图工具

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # 设定图片的默认大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)  # 随机数种子


def sigmoid(Z):
    pass


# 初始化参数：权重矩阵和偏置向量
def initialize_parameters_deep(layer_dims):
    """

    :param layer_dims -- Python 列表，包含网络中每一层的神经元个数
    :return:
    :parameters -- Python 字典，包含参数"W1", "b1", ..., "WL", "bL"
    Wl -- 大小为(layer_dims[l], layer_dims[l-1])的权重矩阵
    bl -- 大小为(layer_dims[l], 1)的偏置向量
    """

    parameters = {}  # 字典
    L = len(layer_dims)  # 网络中的神经元层数
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


# 实现前向传播的线性部分
def linear_forward(A, W, b):
    """
    :param A -- 上一层神经元的输出变量，大小是(layer_dims[l-1],m)
    :param W -- 权重矩阵，是 Numpy 数组，大小是(layer_dims[l],layer_dims[l-1])
    :param b -- 偏置向量，是 Numpy 数组，大小是(layer_dims[l],1)
    :return:
    Z -- 激活函数的输入变量，也称作预激活参数
    cache -- Python 字典，包含"A", "W" and "b"，用于高效地计算反向传播
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


# 实现线性到激活层的前向传播
def linear_activation_forward(A_prev, W, b, activation):
    """
    :param A_prev -- 上一层神经元的输出变量，大小是(layer_dims[l-1],m)
    :param W -- 权重矩阵，是 Numpy 数组，大小是(layer_dims[l],layer_dims[l-1])
    :param b -- 偏置向量，是 Numpy 数组，大小是(layer_dims[l],1)
    :param activation -- 本层使用的激活函数，用"sigmoid"或"relu"表示
    :return:
    A -- 激活函数的输出变量，也被称作后激活值
    cache -- Python 字典，包括 linear_cache 和 activation_cache，用于高效地计算反向传播
    """

    if activation == "sigmoid":
        Z, linear_cache = sigmoid(Z)
