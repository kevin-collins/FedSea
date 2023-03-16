import tensorflow as tf


def attention(queries,
              queries_length,
              keys,
              keys_length,
              scope="attention",
              reuse=None,
              query_masks=None,
              key_masks=None):
    """
    :param queries: A 4d tensor with shape of [?, N, T_q, C]
    :param queries_length: A 2d tensor with shape of [?, N, 1]
    :param keys: A 4d tensor with shape of [?, N, T_k, C]
    :param keys_length: A 3d tensor with shape of [?, N, 1]
    :param scope: Optional scope for `variable_scope`
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name
    :param query_masks: A mask to mask queries with the shape of [?, N, T_k], if query_masks is None, use queries_length to mask queries
    :param key_masks: A mask to mask keys with the shape of [?, N, T_Q],  if key_masks is None, use keys_length to mask keys
    :return: A 4d tensor with shape of (?, N, T_q, C) (?, N, T_q, T_k)
    """

    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.matmul(queries, keys, transpose_b=True)  # (?, N, T_q, T_k)
        query_len = queries.get_shape().as_list()[2]
        key_len = keys.get_shape().as_list()[2]

        # Key Masking
        if key_masks is not None or keys_length is not None:
            if key_masks is None:
                key_masks = tf.sequence_mask(keys_length, key_len)  # (?, N, 1, T_k)
            key_masks = tf.tile(key_masks, [1, 1, query_len, 1])
            paddings = tf.fill(tf.shape(outputs), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
            outputs = tf.where(key_masks, outputs, paddings)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (?, N, T_q, T_k)

        # Query Masking
        if query_masks is None:
            if queries_length is not None:
                queries_length = tf.squeeze(queries_length, axis=2)
                query_masks = tf.sequence_mask(queries_length, query_len)  # (?, N, T_q)

        if query_masks is not None:
            query_masks = tf.tile(tf.expand_dims(query_masks, axis=3), [1, 1, 1, key_len])
            paddings = tf.fill(tf.shape(outputs), tf.constant(0, dtype=tf.float32))
            outputs = tf.where(query_masks, outputs, paddings)

        # Attention vector
        att_vec = outputs

        # Weighted sum (?, N, T_q, T_k) * (?, N, T_k, C)
        outputs = tf.matmul(outputs, keys)  # (?, N, T_q, C)

    return [outputs, att_vec]



class Attention():
    def __init__(self, name):
        #super(Attention, self).__init__('Attention')
        self.name = name

    def __call__(self, queries,
                        queries_length,
                        keys,
                        keys_length,
                        scope="",
                        reuse=None,
                        query_masks=None,
                        key_masks=None):
        if self.name == 'attention':
            return attention(queries, queries_length, keys, keys_length, self.name,
                             reuse, query_masks, key_masks)
        # elif self.name == 'multihead_attention':
        #     return multihead_attention(queries, queries_length, keys, keys_length, self.name,
        #                      reuse, query_masks, key_masks)
        else:
            print('`{}` is not supported. '.format(self.name))
            exit()


