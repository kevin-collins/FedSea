from collections import OrderedDict
from tensorflow.contrib import layers
import tensorflow as tf


class SequenceFeature(object):
    """Construct and mask raw sequence feature"""
    def __init__(self, sequence_block, sequence_length_block, column_builder):
        """
        :param
            "sequence_block": {
                "user_trigger_seq": [
                    "item_id",
                    "shop_id",
                    ...
                ]
            }
        :param
            "sequence_length_block": {
                "user_trigger_seq": "trigger_length"
            }
        """
        self.sequence_block = sequence_block
        self.sequence_length_block = sequence_length_block
        self.seq_column_dict = OrderedDict()
        self.seq_max_length_dict = OrderedDict()
        self.seq_valid_length_dict = OrderedDict()

        # sequence feature
        seq_feature_dict = {}
        for seq_name, att_list in sequence_block.items():
            seq_feature_dict[seq_name] = set(['{}_{}'.format(seq_name, att_name) for att_name in att_list])

        # sequence column
        for seq_name, att_seq_set in seq_feature_dict.items():
            att_column_dict, att_len_dict = column_builder.get_column(seq_name)
            for att_seq_name in att_column_dict:
                if att_seq_name not in att_seq_set:
                    continue
                self.seq_column_dict.update({att_seq_name: att_column_dict[att_seq_name]})
                self.seq_max_length_dict.update({att_seq_name: att_len_dict[att_seq_name]})

        # sequence length column
        for seq_name, length_name in sequence_length_block.items():
            self.seq_valid_length_dict[seq_name] = column_builder.get_column(length_name)

    def concat_seq_features(self, features):
        for seq_name, att_list in self.sequence_block.items():
            for att_name in att_list:
                att_seq_name = '{}_{}'.format(seq_name, att_name)
                att_seq_length = self.seq_max_length_dict[att_seq_name]
                tensor_list = []
                for i in range(att_seq_length):
                    tensor_name = "{}_{}_{}".format(seq_name, i, att_name)
                    tensor_list.append(features[tensor_name])
                try:
                    att_seq_tensor = tf.sparse_concat(sp_inputs=tensor_list, axis=0, expand_nonconcat_dim=True)
                except:
                    att_seq_tensor = tf.concat(values=tensor_list, axis=0)
                features.update({att_seq_name: att_seq_tensor})

    def get_sequence_layer(self, features, sequence_length_layer_dict, scope=None):
        layer_dict = OrderedDict()
        for att_seq_name, column in self.seq_column_dict.items():
            if not isinstance(column, list):
                column = [column]
            layer = layers.input_from_feature_columns(features, column, scope=scope)
            layer = tf.split(layer, self.seq_max_length_dict[att_seq_name], axis=0)  # [?, f_num * f_embedding] * Len
            layer = tf.stack(values=layer, axis=1)  # [?, Len, f_num * f_embedding] or [h * N, T_q, T_k]

            att_seq_length = sequence_length_layer_dict.get(att_seq_name, None)
            if att_seq_length is not None:
                max_length, dim = layer.get_shape().as_list()[1:3]
                masks = tf.sequence_mask(att_seq_length, max_length)
                items_2d = tf.reshape(layer, [-1, tf.shape(layer)[2]])
                layer = tf.reshape(tf.where(tf.reshape(masks, [-1]), items_2d, tf.zeros_like(items_2d)), tf.shape(layer))

            layer_dict[att_seq_name] = layer
        return layer_dict

    def get_sequence_length_layer(self, features, scope=None):
        layer_dict = OrderedDict()
        for seq_name, column in self.seq_valid_length_dict.items():
            layer = tf.to_int32(layers.input_from_feature_columns(features, [column], scope=scope))
            for att_name in self.sequence_block[seq_name]:
                layer_dict['{}_{}'.format(seq_name, att_name)] = layer
        return layer_dict

