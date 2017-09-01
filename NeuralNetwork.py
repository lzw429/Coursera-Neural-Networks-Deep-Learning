import matplotlib as matplotlib
import numpy as np  # Python中科学计算的主要工具
import h5py
import matplotlib.pyplot as plt  # Python作图工具

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # 设定图片的默认大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)  # 随机数种子


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    A = max(0, Z)
    cache = Z
    return A, cache


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


# 实现前向传播的线性部分，即两层神经元之间从 A[l-1] 到 Z[l] 的计算
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


# 实现从线性到激活的前向传播，即从 A[l-1] 到 A[l] 的计算
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
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


# 神经网络的前向传播
def L_model_forward(X, parameters):
    """
    :param X -- Numpy 数组，输入数据
    :param parameters -- 函数initialize_parameters_deep()的输出
    :return:
    最后一层隐藏层到输出层使用sigmoid，隐藏层之间使用relu
    AL -- 后激活值，即神经网络计算出的值
    caches -- 列表，包含每个 linear_relu_forward 的 cache，共有 L-1 个，索引从 0 到 L-2
                      一个 linear_sigmoid_forward 的cache，索引是 L-1
    """
    caches = []
    A = X
    L = len(parameters) // 2  # 双斜杠是floor除法，此处是计算整个神经网络的神经元层数

    for l in range(1, L):  # l从1到L-1
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    """
    :param AL -- 通过计算得到的大小为(1,m)的概率向量
    :param Y -- 大小为(1,m)的标签向量
    :return:
    cost -- 交叉熵代价
    """

    m = Y.shape[1]  # 矩阵Y的列数，即样本数量
    cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    return cost


# 实现从线性到激活的反向传播
def linear_activation_backward(dA, cache, activation):
    """
    :param dA -- 当前层的后激活值的梯度
    :param cache -- 元组(linear_cache, activation_cache)，用于高效地计算反向传播
    :param activation -- 本层使用的激活函数，以文本字符串 "sigmoid" 或 "relu" 的形式存储
    :return:
    dA_prev -- 代价函数J对A_prev求偏导，大小与A_prev相同
    dW -- 代价函数J对W求偏导，
    db --
    """
