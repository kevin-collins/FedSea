import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.framework import ops


def attentional_bias(attention_input_layer, target_input_layer, hidden_size, is_training):
    with tf.variable_scope('diverse_network'):
        W = tf.get_variable(name='W',
                            shape=[attention_input_layer.get_shape().as_list()[1],
                                   target_input_layer.get_shape().as_list()[1]],
                            dtype=tf.float32,
                            collections=[ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES])
        weight = tf.tanh(tf.matmul(attention_input_layer, W), name='weight')
        weighted_target = weight * target_input_layer
        input = tf.concat([attention_input_layer, weighted_target], axis=1)

        for idx, size in enumerate(hidden_size):
            input = layers.fully_connected(
                input,
                size,
                activation_fn=tf.nn.relu,
                scope='layer{}'.format(idx),
                normalizer_fn=layers.batch_norm,
                normalizer_params={'scale': True, 'is_training': is_training}
            )

        logits = layers.fully_connected(
            input,
            1,
            activation_fn=None,
            scope='logits',
        )
        return logits