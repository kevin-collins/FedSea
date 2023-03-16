from collections import OrderedDict
from tensorflow.contrib import layers
import tensorflow as tf


class ActionFeature(object):
    def __init__(self, action_block, column_builder):
        """
        :param 
            "user_action": {
                "item_id": [
                    "item_click",
                    "item_cart",
                    ...
                ],
                "shop_id": [
                ],
                ...
            },
        """
        self.action_column_dict = OrderedDict()
        for key, values in action_block.items():
            key_column = column_builder.get_column(key)
            if isinstance(key_column, list):
                key_column = tuple(key_column)
            self.action_column_dict[key_column] = [column_builder.get_column(value) for value in values]

    def _get_user_action_layer(self, features, scope=None):
        layer_list = []
        for key, values in self.action_column_dict.items():
            if isinstance(key, tuple):
                key = list(key)
            else:
                key = [key]
            key_layer = layers.input_from_feature_columns(features, key, scope=scope)
            for action in values:
                if not isinstance(action, list):
                    action = [action]
                action_layer = layers.input_from_feature_columns(features, action, scope=scope)
                layer_list.append(action_layer * key_layer)
        return tf.concat(layer_list, axis=-1)
