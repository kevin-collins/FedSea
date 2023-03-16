import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import variable_scope


# build deep cross network
# Notice that you have to add variables to tf.GraphKeys.MODEL_VARIABLES if you want to deploy your model on RTP.
# Directly add variables to tf.GraphKeys.MODEL_VARIABLES in tf.get_variables func may occur uninitialized problem when load embedding.
def DCN_network(input_layer, cross_net_depth, is_training, bottleneck_units=None, version="v1"):
    def _build_a_cross_layer(original_input, crossed_input, name_scope):
        with variable_scope.variable_scope(name_scope):
            input_dim = original_input.shape[1]
            weight = tf.get_variable(
                "weight",
                [input_dim, 1],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                # collections=[tf.GraphKeys.MODEL_VARIABLES]
            )
            bias = tf.get_variable(
                "bias",
                [input_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                # collections=[tf.GraphKeys.MODEL_VARIABLES]
            )
            crossed_output = original_input * \
                             tf.matmul(crossed_input, weight) + \
                             bias + crossed_input
        return crossed_output

    def _build_general_cross_layer(
            original_input,
            crossed_input,
            name_scope,
            experts_num=4,
            matrix_rank=16,
            activation_fn=tf.identity):
        with variable_scope.variable_scope(name_scope):
            input_dim = original_input.shape[1]
            weight_U = tf.get_variable(
                "weight_U",
                [1, experts_num, input_dim, matrix_rank],
                initializer=tf.initializers.ones(),
                trainable=True
            )
            weight_V = tf.get_variable(
                "weight_V",
                [1, experts_num, matrix_rank, input_dim],
                initializer=tf.initializers.ones(),
                trainable=True
            )
            weight_C = tf.stack(
                [tf.get_variable(
                    "weight_C_%d" % experts_id,
                    [matrix_rank, matrix_rank],
                    initializer=tf.initializers.identity,
                    trainable=True
                ) for experts_id in range(experts_num)],
                axis=0
            )[None, :, :, :]

            print("weight_U size: ", weight_U.shape)
            print("weight_V size: ", weight_V.shape)
            print("weight_C size: ", weight_C.shape)

            bias = tf.get_variable(
                "bias",
                [1, 1, input_dim],
                initializer=tf.initializers.zeros(),
                trainable=True
            )

            V_mat_x = activation_fn(
                tf.reduce_sum(  # [1, experts_num, matrix_rank, input_dim] * [?, 1, 1, input_dim]
                    weight_V * crossed_input[:, None, None, :],
                    axis=3
                )  # [?, experts_num, matrix_rank]
            )
            print("V_mat_x size: ", V_mat_x.shape)

            C_V_x = activation_fn(
                tf.reduce_sum(  # [1, experts_num, matrix_rank, matrix_rank] * [?, experts_num, 1, matrix_rank]
                    weight_C * V_mat_x[:, :, None, :],
                    axis=3
                )  # [?, experts_num, matrix_rank]
            )
            print("C_V_x size: ", C_V_x.shape)

            U_C_V_x = tf.reduce_sum(  # [1, experts_num, input_dim, matrix_rank] * [?, experts_num, 1, matrix_rank]
                weight_U * C_V_x[:, :, None, :],
                axis=3
            )  # [?, experts_num, input_dim]
            print("U_C_V_x size: ", U_C_V_x.shape)


            experts_weight = layers.fully_connected(
                crossed_input,
                experts_num,
                activation_fn=tf.nn.softmax,
                normalizer_fn=None
            )  # [?, experts_num]
            print("experts_weight size: ", experts_weight.shape)

            # [?, experts_num, input_dim]
            multi_experts_crossed_output = original_input[:, None, :] * (U_C_V_x + bias)
            print("multi_experts_crossed_output size: ", multi_experts_crossed_output.shape)

            mixture_of_multi_experts_crossed_output = tf.reduce_sum(
                experts_weight[:, :, None] * multi_experts_crossed_output,
                axis=1
            )  # [?, input_dim]
            print("mixture_of_multi_experts_crossed_output size", mixture_of_multi_experts_crossed_output.shape)

        return mixture_of_multi_experts_crossed_output


    print("cross_net input size: ", input_layer.shape)

    # network
    crossed_net = input_layer
    for layer_id in range(cross_net_depth):
        if version == "v1":
            crossed_net = _build_a_cross_layer(
                original_input=input_layer,
                crossed_input=crossed_net,
                name_scope="Cross_Net_Layer_%d" % layer_id
            )
        elif version == "v2":
            crossed_net = _build_general_cross_layer(
                original_input=input_layer,
                crossed_input=crossed_net,
                name_scope="Cross_Net_Layer_%d" % layer_id,
                experts_num=4,
                matrix_rank=32,
                activation_fn=tf.tanh
            )


    if bottleneck_units is not None:
        with variable_scope.variable_scope('Cross_Net_Bottleneck_Layer'):
            crossed_net = layers.fully_connected(
                crossed_net,
                bottleneck_units,
                activation_fn=tf.nn.relu,
                normalizer_fn=layers.batch_norm,
                normalizer_params={'scale': True, 'is_training': is_training}
            )

    return crossed_net