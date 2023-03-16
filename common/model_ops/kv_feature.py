from collections import OrderedDict
from tensorflow.contrib import layers
import tensorflow as tf


class KVFeature(object):
    def __init__(self, name_list, column_builder):
        self.kv_column_dict = OrderedDict()
        self.kv_shape_dict = OrderedDict()

        for feature_name in name_list:
            kv_column_info = column_builder.get_column(feature_name)
            self.kv_column_dict[feature_name] = kv_column_info
            self.kv_shape_dict[feature_name] = tf.constant(0, dtype=tf.int32)

    @staticmethod
    def _split_kv_features(sparse_tensor, conf_dict):
        kv_delimiter = conf_dict.get('kv_delimiter', ':')
        value_delimiter = conf_dict.get('value_delimiter', '\004')
        value_dimension = conf_dict.get('value_dimension', 1)
        max_length = conf_dict.get('max_length', 1)

        default_value = '0{}{}'.format(kv_delimiter, value_delimiter.join(['0'] * value_dimension))

        dense_tensor = tf.sparse_to_dense(sparse_tensor.indices,
                                          [sparse_tensor.dense_shape[0], tf.maximum(sparse_tensor.dense_shape[1], 1)],
                                          sparse_tensor.values,
                                          default_value=default_value)
        shape = tf.shape(dense_tensor)

        values = tf.string_split(tf.reshape(dense_tensor, [-1]), kv_delimiter).values
        key, value = tf.split(tf.reshape(values, [-1, 2]), 2, -1)

        if value_dimension > 1:
            value = tf.string_split(tf.squeeze(value, axis=-1), value_delimiter).values
            value = tf.reshape(value, [-1, value_dimension])
        value = tf.string_to_number(value, out_type=tf.float32)

        return key, value, shape

    def get_kv_layer(self, features, scope=None):
        layer_dict = OrderedDict()
        for name, kv_column_info in self.kv_column_dict.items():
            conf_dict = kv_column_info[2]
            sparse_tensor = features[name]
            key_tensor, value_tensor, shape = KVFeature._split_kv_features(sparse_tensor, conf_dict)
            key_layer = layers.input_from_feature_columns({kv_column_info[0][0]: key_tensor}, [kv_column_info[0][1]], scope=scope)
            value_layer = layers.input_from_feature_columns({kv_column_info[1][0]: value_tensor}, [kv_column_info[1][1]], scope=scope)

            last_dim = tf.constant(conf_dict.get('embedding_dimension', 1), dtype=tf.int32)
            key_layer = tf.reshape(key_layer, tf.concat([shape, [last_dim]], axis=0))  # (?, n, embedding_dimension)

            last_dim = tf.constant(conf_dict.get('value_dimension', 1), dtype=tf.int32)
            value_layer = tf.reshape(value_layer, tf.concat([shape, [last_dim]], axis=0))  # (?, n, value_dimension)

            layer_dict[name] = [key_layer, value_layer]
        return layer_dict
