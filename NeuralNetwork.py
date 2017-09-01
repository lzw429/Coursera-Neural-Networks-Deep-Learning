import numpy as np  # Python中科学计算的主要工具

np.random.seed(1)  # 随机数种子


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    A = max(0, Z)
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    :param dA -- 后激活值
    :param cache -- 存储Z，用于高效地计算反向传播
    :return:
    dZ -- 代价函数J对Z的偏导
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    # z<=0 时，dZ = 0
    # z>0 时，dZ = dA
    dZ[Z <= 0] = 0
    return dZ


def sigmoid_backward(dA, cache):
    """
    :param dA -- 后激活值
    :param cache -- 存储Z，用于高效地计算反向传播
    :return:
    dZ -- 代价函数J对Z的偏导
    """

    Z = cache
    s = sigmoid(Z)
    dZ = dA * s * (1 - s)
    return dZ


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
    L = len(layer_dims)  # 网络层数
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


# 为一层实现线性部分的反向传播
def linear_backward(dZ, cache):
    """
    :param dZ -- 代价函数J对Z的偏导
    :param cache -- 来自前向传播的元组(A_prev, W, b)
    :return:
    dA_prev -- 代价函数J对A_prev的偏导，大小与A_prev相同
    dW -- 代价函数J对W的偏导，大小与W相同
    db -- 代价函数J对b的偏导，大小与b相同
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]  # 样本个数

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


# 实现从线性到激活的反向传播
def linear_activation_backward(dA, cache, activation):
    """
    :param dA -- 当前层的后激活值的梯度
    :param cache -- 元组(linear_cache, activation_cache)，用于高效地计算反向传播
    :param activation -- 本层使用的激活函数，以文本字符串 "sigmoid" 或 "relu" 的形式存储
    :return:
    dA_prev -- 代价函数J对A_prev的偏导，大小与A_prev相同
    dW -- 代价函数J对W的偏导，大小与W相同
    db -- 代价函数J对b的偏导，大小与b相同
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    :param AL -- 概率向量，前向传播的输出
    :param Y -- 标签向量
    :param caches -- 列表，包含每个 linear_relu_forward 的 cache，共有 L-1 个，索引从 0 到 L-2
                             一个 linear_sigmoid_forward 的cache，索引是 L-1
    :return:
    grads -- 存储各项梯度的字典
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """

    grads = {}
    L = len(caches)  # 网络层数
    m = AL.shape[1]  # 样本个数
    Y = Y.reshape(AL.shape)  # Y 与 AL 维度一致

    # 初始化反向传播
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # 第L层的反向传播：sigmoid 到线性的梯度
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                  current_cache,
                                                                                                  "sigmoid")
    for l in reversed(range(L - 1)):  # l从L-2 到0
        # relu 到线性的梯度
        current_cache = caches[l]
        dA_prev_l, dW_l, db_l = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_l
        grads["dW" + str(l + 1)] = dW_l
        grads["db" + str(l + 1)] = db_l
    return grads


# 实现前向传播与反向传播后，能计算出迭代需要的梯度，下面对参数进行迭代
def update_parameters(parameters, grads, learning_rate):
    """
    :param parameters -- Python 字典
    :param grads -- Python 字典
    :param learning_rate -- 学习率
    :return:
    parameters -- Python 字典，包含更新后的参数
    """

    L = len(parameters) // 2  # 网络层数

    for l in range(L):  # l从0 到L-1
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters
