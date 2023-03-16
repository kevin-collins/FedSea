# -*- coding: utf-8 -*-
import copy
import uuid
import numpy as np
import tensorflow as tf


def ltr_relu(inputs):

    # Define the op in python
    def _relu(x):
        return np.maximum(x, 0.)

    # Define the op's gradient in python
    def _relu_grad(x, grad):
        alpha = 0.01
        limit = -0.01
        g = np.float32(x > 0)
        g[grad < 0] = 1
        g[np.bitwise_and(grad > 0, np.bitwise_and(x < 0, x > limit))] = alpha
        return g

    # An adapter that defines a gradient op compatible with Tensorflow
    def _relu_grad_op(op, grad):
        x = op.inputs[0]
        x_grad = grad * tf.py_func(_relu_grad, [x, grad], tf.float32)
        return x_grad

    # Register the gradient with a unique id
    grad_name = "LtrReluGrad_" + str(uuid.uuid4())
    tf.RegisterGradient(grad_name)(_relu_grad_op)
    # Override the gradient of the custom op
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": grad_name}):
        output = tf.py_func(_relu, [inputs], tf.float32)
    return output


def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1),
                                 collections=[tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES])
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))


def get_act_fn(name):
    activations = {
        'relu': tf.nn.relu,
        'lrelu': lambda x: tf.nn.leaky_relu(x, alpha=0.01),
        'tanh': tf.tanh,
        'sigmoid': tf.sigmoid,
        'softmax': tf.nn.softmax,
        'elu': tf.nn.elu,
        'softplus': tf.nn.softplus,
        'ltr_relu': ltr_relu,
        'prelu': prelu,
        'mish': mish,
        'identity': tf.identity
    }
    return activations[name]


# specific gated embedding
def get_valid_embedding(x, part=0, valid_dimension=8, embedding_dimension=17):
    shape = x.shape.as_list()
    if shape[-1] != embedding_dimension:
        x = tf.reshape(x, [-1] + shape[1:-1] + [shape[-1] // embedding_dimension, embedding_dimension])
    _x = x[..., :valid_dimension]
    _y = x[..., valid_dimension:-1]
    _g = x[..., -1:]

    if part == 0:
        z = tf.reshape(_x, [-1] + shape[1:-1] + [shape[-1] // embedding_dimension * valid_dimension])
    elif part == 1:
        z = tf.stop_gradient(_x) * tf.sigmoid(_g) + _y
        z = tf.reshape(z, [-1] + shape[1:-1] + [shape[-1] // embedding_dimension * valid_dimension])
    else:
        raise Exception('unsupported part value:{}'.format(part))
    return z


def get_part_embedding(x, part=0, valid_dimension=8, embedding_dimension=16, stop_gradient=False, isolate=False, aux_reg=False):
    shape = x.shape.as_list()
    if shape[-1] != embedding_dimension:
        x = tf.reshape(x, [-1] + shape[1:-1] + [shape[-1] // embedding_dimension, embedding_dimension])
    _x = x[..., :valid_dimension]
    _y = x[..., valid_dimension:2*valid_dimension]

    if part == 0:
        z = tf.reshape(_x, [-1] + shape[1:-1] + [shape[-1] // embedding_dimension * valid_dimension])
    elif part == 1:
        if isolate:
            z = _y
        else:
            if stop_gradient:
                z = tf.stop_gradient(_x) + _y
            else:
                z = _x + _y
        z = tf.reshape(z, [-1] + shape[1:-1] + [shape[-1] // embedding_dimension * valid_dimension])
    else:
        raise Exception('unsupported part value:{}'.format(part))

    if aux_reg:
        aux_loss = tf.reduce_sum(tf.square(tf.stop_gradient(_x) - _y), axis=range(len(_x.shape.as_list()))[1:])
        tf.add_to_collection('AUX_LOSS_COLLECTION', aux_loss)
    return z


def get_gaussian_embedding(x, is_training, valid_dimension=8, embedding_dimension=16):
    shape = x.shape.as_list()
    if shape[-1] != embedding_dimension:
        x = tf.reshape(x, [-1] + shape[1:-1] + [shape[-1] // embedding_dimension, embedding_dimension])
    _u = x[..., :valid_dimension]
    _rho = x[..., valid_dimension:2*valid_dimension]

    def _sample_fn():
        epsilon = tf.truncated_normal(shape=[valid_dimension])
        return _u + tf.log1p(tf.exp(_rho)) * epsilon

    z = tf.cond(is_training,
                lambda: _sample_fn(),
                lambda: _u)
    z = tf.reshape(z, [-1] + shape[1:-1] + [shape[-1] // embedding_dimension * valid_dimension])
    return z


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    多核或单核高斯核矩阵函数，根据输入样本集x和y，计算返回对应的高斯核矩阵
    Params:
     source: (b1,n)的X分布样本数组
     target:（b2，n)的Y分布样本数组
     kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定，如果固定，则为单核MMD
    Return:
      sum(kernel_val): 多个核矩阵之和
    '''
    # 堆叠两组样本，上面是X分布样本，下面是Y分布样本，得到（b1+b2,n）组总样本
    n_samples = tf.to_float(tf.shape(source)[0] + tf.shape(target)[0], name='ToFloat')
    total = tf.concat((source, target), axis=0)
    # 对总样本变换格式为（1,b1+b2,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按行复制
    total0 = tf.expand_dims(total, axis=0)
    total0 = tf.broadcast_to(total0, [tf.shape(total)[0], tf.shape(total)[0], tf.shape(total)[1]])
    # 对总样本变换格式为（b1+b2,1,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按列复制
    total1 = tf.expand_dims(total, axis=1)
    total1 = tf.broadcast_to(total1, [tf.shape(total)[0], tf.shape(total)[0], tf.shape(total)[1]])
    # total1 - total2 得到的矩阵中坐标（i,j, :）代表total中第i行数据和第j行数据之间的差
    # sum函数，对第三维进行求和，即平方后再求和，获得高斯核指数部分的分子，是L2范数的平方
    L2_distance_square = tf.cumsum(tf.square(total0 - total1), axis=2)

    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # need float32
        bandwidth = tf.reduce_sum(L2_distance_square) / (n_samples ** 2 - n_samples)
    # 多核MMD
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # 高斯核函数的数学表达式
    kernel_val = [tf.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # 多核合并

def mk_mmd_loss(source, target, batch_size = 50,kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
     source: (b1,n)的X分布样本数组
     target:（b2，n)的Y分布样本数组
     kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定，如果固定，则为单核MMD
 Return:
     loss: MK-MMD loss
    '''
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 将核矩阵分成4部分
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    # 这里计算出的n_loss是每个维度上的MK-MMD距离，一般还会做均值化处理
    n_loss = loss / float(batch_size)
    # kernels = guassian_kernel(source, target,
    #                            kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # loss = 0
    # loss += kernels[0, 1] + kernels[1, 2]
    # loss -= kernels[0, 2] + kernels[1, 1]
    return tf.maximum(tf.reduce_mean(n_loss),0.)

def random_shuffle(data,label,label2=None):
    randnum = np.random.randint(0, 1234)
    data = tf.gather(data, tf.random.shuffle(tf.range(tf.shape(data)[0]),seed=randnum) )
    label = tf.gather(label, tf.random.shuffle(tf.range(tf.shape(label)[0]),seed=randnum) )

    if label2 is not None:
        label2 = tf.gather(label2, tf.random.shuffle(tf.range(tf.shape(label2)[0]), seed=randnum))
        return data, label,label2
    # data = tf.random_shuffle(data,seed=randnum)
    # label = tf.random_shuffle(label,seed=randnum)
    return data,label

def caculate_model_distance(sess,scope1,scope2):
    vars1 = sess.run(tf.trainable_variables(scope=scope1))
    vars2 = sess.run(tf.trainable_variables(scope=scope2))

    assert len(vars1)==len(vars2)

    norm = 0.
    for v1,v2 in zip(vars1,vars2):
        norm += tf.norm(v1-v2)

    return norm

def get_mdl_params(model_scopes,n_par=None):
    if n_par is None:
        n_par = 0
        for param in tf.trainable_variables(scope=model_scopes[0]):
            n_par +=tf.reshape(param,[-1]).shape[0]

    param_mats = []

    for i,model_scope in enumerate(model_scopes):
        param_mat = None
        for param in tf.trainable_variables(scope=model_scope):
            temp = tf.reshape(param,[-1])
            if param_mat is None:
                param_mat = temp
            else:
                param_mat = tf.concat([param_mat,temp],axis=0)

        param_mats.append(param_mat)
    return param_mats

def set_client_from_params(model_scope, params,sess):
    idx = 0
    for i,param in enumerate(tf.trainable_variables(scope=model_scope)):
        length = tf.reshape(param,[-1]).shape[0]
        data = sess.run(tf.reshape(params[idx:idx+length],param.shape))
        param.load(data,sess)
        idx += length

def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        #pylint: disable=W0212
        session = session._sess
    return session
