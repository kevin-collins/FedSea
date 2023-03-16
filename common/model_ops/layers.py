from tensorflow.contrib import layers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import tensorflow as tf


def feed_forward_net(net,
                     hidden_units,
                     activation_fn=tf.nn.relu,
                     kernel_initializer=None,
                     kernel_regularizer=None,
                     dropout=None,
                     dnn_parent_scope=None,
                     is_training=True):

    for layer_id, num_hidden_units in enumerate(hidden_units):
        with tf.variable_scope(dnn_parent_scope + '_h_%d' % layer_id, values=(net,)) as scope:
            net = tf.layers.dense(net,
                                  units=num_hidden_units,
                                  activation=activation_fn,
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer,
                                  name=scope)
            if (layer_id+1)<len(hidden_units):
                net = layers.batch_norm(net, is_training=is_training)
            if dropout is not None:
                keep_prob = 1-dropout
                net = layers.dropout(net, keep_prob=keep_prob, is_training=is_training)
    return net


def multi_fully_connected(inputs,
                          num_outputs,
                          activation_fn=nn.relu,
                          normalizer_fn=None,
                          normalizer_params=None,
                          weights_initializer=initializers.xavier_initializer(),
                          biases_initializer=init_ops.zeros_initializer(),
                          variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES],
                          scope=None):
    input_shape = inputs.get_shape().as_list()
    assert len(input_shape) > 2, 'shape of inputs must larger than 2'
    inputs = tf.expand_dims(inputs, axis=-1)
    with tf.variable_scope(name_or_scope=scope):
        weigths = tf.get_variable(name='weights',
                                  shape=[1] + input_shape[1:] + [num_outputs],
                                  dtype=tf.float32,
                                  trainable=True,
                                  collections=variables_collections,
                                  initializer=weights_initializer)
        biases = tf.get_variable(name='biases',
                                 shape=[1] + input_shape[1:-1] + [num_outputs],
                                 dtype=tf.float32,
                                 trainable=True,
                                 collections=variables_collections,
                                 initializer=biases_initializer)
        outputs = tf.reduce_sum(tf.multiply(inputs, weigths), axis=-2) + biases
        output_shape = outputs.get_shape().as_list()
        outputs = tf.reshape(outputs, [-1, np.prod(output_shape[1:])])
        # Apply normalizer function / layer.
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        if normalizer_fn is not None:
            if not normalizer_params:
                normalizer_params = {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        outputs = tf.reshape(outputs, shape=[-1] + output_shape[1:])
        return outputs
