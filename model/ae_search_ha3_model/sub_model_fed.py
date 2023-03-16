# -*- coding: utf-8 -*-
from __future__ import print_function
from collections import OrderedDict

import utils.util
from common.base.mode import ModeKeys
from common.model_ops.layers import multi_fully_connected
from utils.util import get_act_fn, get_part_embedding
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope
from common.model_ops import ops as base_ops
# from common.model_ops.attention import multihead_attention
import tensorflow as tf
import numpy as np

from common.model_ops.attention import AttentionFeature
from common.model_ops.sequence import SequenceFeature
from model.ae_search_ha3_model.action import ActionFeature
from model.ae_search_ha3_model.query import Query
from common.fg.feature_column_builder import FeatureColumnBuilder
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import dtypes
from utils.util import mk_mmd_loss
from utils.util import get_mdl_params,set_client_from_params
class GradientReversal(object):
    def __init__(self, name="GradRevIdentity"):
        self.call_num = 0  # 用于防止多次调用call函数时, 名字被重复使用
        self.name = name

    def call(self, x, s=1.0):
        op_name = self.name + "_" + str(self.call_num)
        self.call_num += 1

        @tf.RegisterGradient(op_name)
        def reverse_grad(op, grad):
            return [-grad * s]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": op_name}):  # 将下面的identity的梯度改成op_name对应的梯度计算方式
            y = tf.identity(x)
        return y

    def __call__(self, x, s=1.0):
        return self.call(x, s)



class SubModel(object):

    def __init__(self, co_model, name, *args, **kwargs):
        self.co_model = co_model
        self.name = name

        if self.name != 'Global':
            self.country_idx = self.co_model.FLAGS.countrys.index(name)
        else:
            self.country_idx = len(self.co_model.FLAGS.countrys)

        self.labels = None
        self.input_dict = None
        self.loss = None
        self.logits = None
        self.predicts = None
        self.train_op = None
        self.auc = None
        self.auc_update = None

        self.cnty_weight = None
        self.cnty_label = None
        self.cnty_loss = None
        self.aux_loss = None

        self.branch_hidden_units = [64]

        self.save_feature = False
        self.country_id = None
        self.features = None

        self.ctr_sample_id = None
        self.ctr_real_label = None

        self.user_net_stop = None
        self.dense_input_layer_stop = None

        # 防止分布式下重名 KeyError:Registering two gradient with name 'GradRevIdentity_0'
        self.gr = GradientReversal(name=self.name)

        ######################################################################################################################
        #                                                                                                                    #
        #                                               Feature Column                                                       #
        #                                                                                                                    #
        ######################################################################################################################
        # use local embedding

        with variable_scope.variable_scope('{}_SubModel'.format(self.name),
                                           partitioner=base_ops.partitioner(ps_num=self.co_model.ps_num, mem=self.co_model.embedding_partition_size)):
            # get embedding for each column list
            self.column_builder = FeatureColumnBuilder(self.co_model.FLAGS, self.co_model.FLAGS.fg_conf, self.co_model.FLAGS.fc_conf)

            with variable_scope.variable_scope('User'):
                # feature column
                # order by self.co_model.user_sparse_features
                self.user_sparse_column = self.column_builder.get_column_list(self.co_model.user_sparse_features)
                self.user_dense_column = self.column_builder.get_column_list(self.co_model.user_dense_features)
                self.user_behavior_column = self.column_builder.get_column_list(self.co_model.user_behavior_features)

            with variable_scope.variable_scope('Item'):
                self.item_sparse_column = self.column_builder.get_column_list(self.co_model.item_sparse_features)
                self.item_dense_column = self.column_builder.get_column_list(self.co_model.item_dense_features)
                self.item_behavior_column = self.column_builder.get_column_list(self.co_model.item_behavior_features)

                # price level column
                self.origin_item_price_level_column = [
                    layers.real_valued_column(column_name='origin_item_price_level', dimension=1, default_value=0.0)]

            ######################################################################################################################
            #                                                                                                                    #
            #                                      Sequence & Attention & Query & Action Feature                                 #
            #                                                                                                                    #
            ######################################################################################################################
            with variable_scope.variable_scope('User'):
                # self.sequence = SequenceFeature(self.co_model.column_conf['sequence_block'], self.co_model.column_conf['sequence_length_block'],
                #                                 self.column_builder)
                # self.attention = AttentionFeature(self.co_model.column_conf['attention_block'], self.column_builder)
                self.query = Query(self.column_builder)
                self.user_action = ActionFeature(self.co_model.column_conf['user_action'], self.column_builder)
                self.query_action = ActionFeature(self.co_model.column_conf['query_action'], self.column_builder)

        with tf.variable_scope(name_or_scope='TRAIN_STEP', reuse=tf.AUTO_REUSE):
            self.train_step = tf.get_variable(name='{}_Train_Step'.format(name),
                                              shape=(),
                                              dtype=tf.int32,
                                              trainable=False,
                                              collections=[tf.GraphKeys.GLOBAL_VARIABLES],
                                              initializer=tf.zeros_initializer)

    def build_inputs(self, fg_features, batch_labels):
        with tf.variable_scope(name_or_scope='{}_SubModel'.format(self.name)):
            with tf.name_scope('Input_Pipeline'):
                print(fg_features)
                self.country_id = fg_features['feature_3']
                self.country_id = tf.sparse_to_dense(self.country_id.indices, self.country_id.dense_shape, self.country_id.values,
                                              default_value='Other')
                self.sample_nums = tf.shape(self.country_id)[0]

                if self.co_model.FLAGS.FedSFA:
                    self.dis_label = tf.one_hot(tf.fill([self.sample_nums],self.country_idx),len(self.co_model.FLAGS.countrys))

                # self.sequence.concat_seq_features(fg_features)
                self.query.update_query_length(fg_features)
                self.column_builder.update_multi_hash_features(fg_features)

                SubModel._update_item_price(fg_features)
                self.labels = batch_labels

            with tf.variable_scope(name_or_scope='Input_Column', reuse=False,
                                   partitioner=base_ops.partitioner(ps_num=self.co_model.ps_num,
                                                                    mem=self.co_model.embedding_partition_size),
                                   ):
                self.input_dict = self._build_input_layer(fg_features)

    def _build_input_layer(self, fg_features):
        part = 0
        aux_reg = False

        # print(self.name, tf.trainable_variables(scope='{}_SubModel'.format(self.name)))
        with variable_scope.variable_scope('User'):
            # tensor
            user_sparse_input_layer = layers.input_from_feature_columns(fg_features, self.user_sparse_column)
            user_dense_input_layer = layers.input_from_feature_columns(fg_features, self.user_dense_column)
            user_behavior_input_layer = layers.input_from_feature_columns(fg_features, self.user_behavior_column)

            item_sparse_input_layer = layers.input_from_feature_columns(fg_features, self.item_sparse_column)
            item_dense_input_layer = layers.input_from_feature_columns(fg_features, self.item_dense_column)
            item_behavior_input_layer = layers.input_from_feature_columns(fg_features, self.item_behavior_column)

            # ******  modify to not use share embedding
            query_token_input_layer = self.query.get_specific_query_token_layer(fg_features, method='part', part=part, isolate=True,
                                                                                         valid_dimension=8, embedding_dimension=16, aux_reg=aux_reg)
            # sequence_length_layer_dict = self.sequence.get_sequence_length_layer(fg_features,)
            # sequence_input_layer_dict = self.sequence.get_sequence_layer(fg_features, sequence_length_layer_dict)

        with variable_scope.variable_scope('Item'):
            item_price_level = layers.input_from_feature_columns(fg_features, self.origin_item_price_level_column) / 20.0

        # print(self.name, tf.build_model_with_part_featuretrainable_variables(scope='{}_SubModel'.format(self.name)))

        with tf.variable_scope(name_or_scope='Input_BN'):
            with variable_scope.variable_scope('Item'):
                item_dense_input_layer = layers.batch_norm(tf.nn.relu(item_dense_input_layer),
                                                           scale=True, is_training=self.co_model.is_training)
                item_behavior_input_layer = tf.log1p(tf.nn.relu(item_behavior_input_layer))
                item_behavior_input_layer = layers.batch_norm(item_behavior_input_layer,
                                                              scale=True, is_training=self.co_model.is_training)

            with variable_scope.variable_scope('User'):
                user_dense_input_layer = layers.batch_norm(tf.nn.relu(user_dense_input_layer),
                                                           scale=True, is_training=self.co_model.is_training)
                user_behavior_input_layer = tf.log1p(tf.nn.relu(user_behavior_input_layer))
                user_behavior_input_layer = layers.batch_norm(user_behavior_input_layer,
                                                              scale=True, is_training=self.co_model.is_training)

        return {
            'user_sparse_input_layer': user_sparse_input_layer,
            'user_dense_input_layer': user_dense_input_layer,
            'user_behavior_input_layer': user_behavior_input_layer,
            'item_sparse_input_layer': item_sparse_input_layer,
            'item_dense_input_layer': item_dense_input_layer,
            'item_behavior_input_layer': item_behavior_input_layer,
            'query_token_input_layer': query_token_input_layer,
            # 'sequence_length_layer_dict': sequence_length_layer_dict,
            # 'sequence_input_layer_dict': sequence_input_layer_dict,
            'item_price_level': item_price_level,
        }



    def build_model(self):

        self.build_model_select_feature_map()

        if self.co_model.FLAGS.FedSFA:
            # ******************** froward dis
            map_feature = self.maped_feature

            self.map_feature = map_feature

            with variable_scope.variable_scope('Map_Net', reuse=tf.AUTO_REUSE):
                # reverse gradient
                map_feature_reverse = self.gr(map_feature, self.co_model.runner.p)
                # map_feature_reverse = map_feature
                for layer_id, num_hidden_units in enumerate([64,128,len(self.co_model.FLAGS.countrys)]):
                    with variable_scope.variable_scope('Hidden_{}'.format(layer_id)):
                        map_feature_reverse = layers.fully_connected(map_feature_reverse,
                                                                     num_hidden_units,
                                                                     activation_fn=get_act_fn(
                                                                         self.co_model.dnn_hidden_units_act_op[
                                                                             layer_id]),
                                                                     )
                self.dis_logits = map_feature_reverse


    def build_model_select_feature_map(self):

        with variable_scope.variable_scope('Map_Net', reuse=tf.AUTO_REUSE):
            with variable_scope.variable_scope('Feature_Select_Net'):
                p = tf.get_variable(name="mask_global", shape=[64],
                                    initializer=tf.random_normal_initializer(mean=0, stddev=0.001))
                self.mask_global = tf.sigmoid(tf.constant([10.] * 64) * p)


        with arg_scope(base_ops.model_arg_scope(weight_decay=self.co_model.dnn_l2_reg)):

            with tf.variable_scope(name_or_scope='{}_SubModel'.format(self.name),
                               partitioner=base_ops.partitioner(ps_num=self.co_model.ps_num, mem=self.co_model.embedding_partition_size)):
                    with variable_scope.variable_scope('model'):
                        # *********************************** UserQuery **************************************
                        with variable_scope.variable_scope('User'):
                            with variable_scope.variable_scope('UserQuery_Sparse_Projection'):
                                query_shape = self.input_dict['query_token_input_layer'].shape.as_list()
                                query_token_input_layer = tf.reshape(self.input_dict['query_token_input_layer'],
                                                                     [-1, query_shape[1] * query_shape[2]])

                                user_sparse_input_layer = layers.fully_connected(
                                    self.input_dict['user_sparse_input_layer'],
                                    256,
                                    activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[0]),
                                )
                                sparse_input_layer = tf.concat([
                                    user_sparse_input_layer,
                                    query_token_input_layer,
                                ], axis=-1)
                                sparse_input_layer = layers.batch_norm(sparse_input_layer, scale=True,
                                                                       is_training=self.co_model.is_training)

                            with variable_scope.variable_scope('UserQuery_Dense_Projection'):
                                dense_input_layer = tf.concat([self.input_dict['user_dense_input_layer'],
                                                               self.input_dict['user_behavior_input_layer'],
                                                               ], axis=-1)

                                dense_input_layer = layers.fully_connected(
                                    dense_input_layer,
                                    256,
                                    activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[0]),
                                )
                                dense_input_layer = layers.batch_norm(dense_input_layer, scale=True,
                                                                      is_training=self.co_model.is_training)

                            with variable_scope.variable_scope('UserQuery_Net'):
                                user_net = layers.fully_connected(
                                    tf.concat([sparse_input_layer, dense_input_layer], axis=1),
                                    self.co_model.user_output_size,
                                    activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[0])
                                )
                                self.user_net = tf.identity(user_net, 'Output')
                        # *********************************** Item **************************************
                        with variable_scope.variable_scope('Item'):
                            with variable_scope.variable_scope('Item_Sparse_Projection'):
                                item_sparse_input_layer = layers.fully_connected(
                                    self.input_dict['item_sparse_input_layer'],
                                    256,
                                    activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[0]),
                                )
                                sparse_input_layer = layers.batch_norm(item_sparse_input_layer, scale=True, is_training=self.co_model.is_training)

                            with variable_scope.variable_scope('Item_Dense_Projection'):
                                dense_input_layer = tf.concat([
                                                               self.input_dict['item_dense_input_layer'],
                                                               self.input_dict['item_behavior_input_layer'],
                                                               ], axis=-1)
                                dense_input_layer = layers.fully_connected(
                                    dense_input_layer,
                                    256,
                                    activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[0]),
                                )
                                dense_input_layer = layers.batch_norm(dense_input_layer, scale=True, is_training=self.co_model.is_training)

                            with variable_scope.variable_scope('Item_Net'):
                                item_net = layers.fully_connected(
                                        tf.concat([sparse_input_layer, dense_input_layer], axis=1),
                                        self.co_model.item_output_size,
                                        activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[0])
                                    )
                                self.item_net = tf.identity(item_net, name="Output")

                        with variable_scope.variable_scope("Feature_Net"):
                            net = tf.concat([self.user_net, self.item_net], axis=1)
                            self.ori_feature = self.mask_global * tf.stop_gradient(net)
                            self.feature_before = net

                            labels = tf.reshape(self.labels, [-1])

                            # sort mask
                            index = tf.contrib.framework.argsort(self.mask_global, direction='DESCENDING')

                            # select non-iid feature(global_net)
                            global_index = tf.contrib.framework.sort(index[:32],axis=0)
                            local_index = tf.contrib.framework.sort(index[32:],axis=0)
                            global_net = tf.gather(net, global_index, axis=1)
                            local_net = tf.gather(net, local_index, axis=1)

                            with variable_scope.variable_scope("Transform_Feature_Net"):
                                self.transform_scale = tf.get_variable(name="transform_scale_vector",shape=[32],
                                                                    initializer=tf.ones_initializer)
                                self.transform_bias = tf.get_variable(name="transform_bias_vector",shape=[32],
                                                                   initializer=tf.zeros_initializer)

                                self.global_feature = tf.multiply(self.transform_scale, global_net) + self.transform_bias
                                self.global_feature_map = tf.multiply(self.transform_scale, tf.stop_gradient(global_net)) + self.transform_bias

                            self.local_feature = local_net
                            self.local_feature_map = tf.stop_gradient(local_net)

                            # for predict
                            indices = tf.concat(
                                [tf.expand_dims(global_index, axis=1), tf.expand_dims(local_index, axis=1)], axis=0)
                            updates = tf.transpose(tf.concat([self.global_feature, self.local_feature], axis=1))
                            net = tf.transpose(tf.scatter_nd(indices, updates, tf.shape(updates)))
                            self.net = net
                            self.feature_after = net

                            # stop gradient for dis
                            updates_map = tf.transpose(tf.concat([self.global_feature_map, self.local_feature_map], axis=1))
                            self.maped_feature = tf.stop_gradient(self.mask_global) * tf.transpose(tf.scatter_nd(indices, updates_map, tf.shape(updates_map)))
                            # pass gradient
                            # self.maped_feature = tf.stop_gradient(self.mask_global) * net

                        with tf.variable_scope(name_or_scope='DNN_Network'):
                                for layer_id, num_hidden_units in enumerate(self.co_model.dnn_hidden_units):
                                    with variable_scope.variable_scope('Hidden_{}'.format(layer_id)):
                                        net = layers.fully_connected(net,
                                                                     num_hidden_units,
                                                                     activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[layer_id]),
                                                                     )
                                        net = layers.batch_norm(net, scale=True, is_training=self.co_model.is_training)

                                        if self.co_model.need_dropout and self.co_model.dropout_rate > 0:
                                            net = tf.layers.dropout(net, rate=self.co_model.dropout_rate, training=self.co_model.is_training)

                                self.logits = net


    def build_loss(self):
        self.predicts = tf.sigmoid(tf.clip_by_value(self.logits, -20., 20.))

        with tf.name_scope('Loss'):
            self.predict_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            )
            self.loss = self.predict_loss
            l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            if self.co_model.FLAGS.FedSFA:

                self.dis_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.dis_label, logits=self.dis_logits)
                )
                self.loss += self.dis_loss

    def build_optimizer(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.co_model.learning_rate) # for TF1120
        update_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name.upper() in v.name.upper()]
        with tf.control_dependencies(update_ops):

            if self.co_model.FLAGS.FedSFA:
                self_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{}_SubModel'.format(self.name))
                self.train_op = optimizer.minimize(self.loss, global_step=self.train_step, var_list=self_vars)
            else:
                self.train_op = optimizer.minimize(self.loss, global_step=self.train_step)


    def build_summary(self):
        with tf.name_scope('Metrics/{}'.format(self.name)):
            if self.co_model.FLAGS.mode == ModeKeys.LOCAL:
                self.auc, self.auc_update = tf.metrics.auc(labels=self.labels,
                                                           predictions=self.predicts,
                                                           num_thresholds=2000)
            else:
                # worker_device = '/job:worker/task:{}'.format(self.co_model.FLAGS.task_index)
                # with tf.device(worker_device):
                self.auc, self.auc_update = tf.metrics.auc(labels=self.labels,
                                                           predictions=self.predicts,
                                                           num_thresholds=2000)


        with tf.name_scope('Summary/{}'.format(self.name.upper())):
            self.label_mean = tf.reduce_mean(self.labels)
            tf.summary.scalar(name='AUC', tensor=self.auc)
            tf.summary.scalar(name='Loss', tensor=self.loss)
            # tf.summary.scalar(name='Cnty_Loss', tensor=self.cnty_loss)
            tf.summary.scalar(name='Label_Mean', tensor=tf.reduce_mean(self.labels))
            tf.summary.scalar(name='Predict_Mean', tensor=tf.reduce_mean(self.predicts))

            if self.co_model.FLAGS.FedSFA:
                tf.summary.scalar(name='dis_Loss',tensor=self.dis_loss)

            if self.co_model.FLAGS.FedSFA:
                tf.summary.histogram(name='mask_global', values=self.mask_global)

            if self.name.upper() == 'CTR':
                tf.summary.scalar(name='AUX_Loss', tensor=self.aux_loss)


    @staticmethod
    def _update_item_price(features):
        item_price_level = features['feature_55']
        features.update({'origin_item_price_level': item_price_level})

    def _create_cnty_dense_feature(self,
                               cnty_item_dense_input_layer,
                               country_id_input_layer,
                               gate_num,
                               hidden_size):
        net = cnty_item_dense_input_layer
        for i, size in enumerate(hidden_size):
            net = layers.fully_connected(
                net,
                size,
                scope='cnty_ft_layer_{}'.format(i),
                activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[0]),
            )
            net = layers.batch_norm(net, scale=True, is_training=self.co_model.is_training)

        cnty_embed_weight = layers.fully_connected(
            country_id_input_layer,
            gate_num,
            scope='cnty_embed_weight',
            activation_fn=None,
            normalizer_fn=None
        )

        with tf.name_scope('Summary/{}'.format(self.name.upper())):
            img = tf.reduce_mean(cnty_embed_weight, axis=0, keep_dims=True)
            tf.summary.histogram('SoftmaxGate', img)
            img = tf.reshape(img, [1, 1, -1, 1])
            tf.summary.image('CountryGate', img)

        return net, cnty_embed_weight

    def _create_multi_branch_network(self,
                                 net,
                                 cnty_embed,
                                 cnty_weight,
                                 gate_num,
                                 hidden_size):
        net = tf.expand_dims(net, axis=1)
        net = tf.tile(net, [1, gate_num, 1])
        # net = multi_fully_connected(net,
        #                             128,
        #                             activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[0]),
        #                             normalizer_fn=layers.batch_norm,
        #                             normalizer_params={'scale': True, 'is_training': self.co_model.is_training},
        #                             scope='cnty_tile')

        cnty_gate = tf.expand_dims(tf.nn.softmax(cnty_weight), axis=-1)
        cnty_embed = tf.expand_dims(cnty_embed, axis=1) * cnty_gate

        layer = tf.concat([net, cnty_embed], axis=-1, name='cnty_layer_0')
        for i, size in enumerate(hidden_size):
            layer = multi_fully_connected(
                layer,
                size,
                activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[0]),
                scope='cnty_layer_{}'.format(i + 1),
                normalizer_fn=layers.batch_norm,
                normalizer_params={'scale': True, 'is_training': self.co_model.is_training}
            )

        cnty_logit = multi_fully_connected(
            layer,
            1,
            activation_fn=None,
            scope='cnty_logit',
        )
        cnty_logit = tf.reduce_sum(cnty_logit * cnty_gate, axis=1)
        return cnty_logit


    # def _create_attention_network(self,
    #                           attention_input_layer_dict,
    #                           sequence_input_layer_dict,
    #                           sequence_length_layer_dict,
    #                           query_token_input_layer,
    #                           units=16):
    #     common_lst = []
    #     common_layer = None
    #     for name, layer in sequence_input_layer_dict.items():
    #         if name not in attention_input_layer_dict:
    #             common_lst.append(layer)
    #     if common_lst:
    #         common_layer = tf.add_n(common_lst)
    #
    #     seq_layer_list, query_layer_list, key_masks = [], [], None
    #     for name, query_layers in attention_input_layer_dict.items():
    #         seq_layer = sequence_input_layer_dict[name]
    #         seq_length = sequence_length_layer_dict.get(name, None)
    #         if seq_length is None:
    #             raise Exception('keys {} length is None'.format(name))
    #         if common_lst and name.endswith('cate_id'):  # same cate_id within the whole sequence
    #             seq_layer += common_layer
    #         if len(query_layers) > 1:
    #             query_layer = tf.concat(query_layers, axis=1)
    #         else:
    #             query_layer = query_layers[0]
    #
    #         if key_masks is None:
    #             key_len = seq_layer.get_shape().as_list()[1]
    #             key_masks = tf.sequence_mask(seq_length, key_len)
    #
    #         seq_layer_list.append(seq_layer)
    #         query_layer_list.append(query_layer)
    #
    #     seq_layer = tf.concat(seq_layer_list, axis=-1)
    #     query_layer = tf.concat(query_layer_list, axis=-1)
    #     token_layer = tf.reshape(query_token_input_layer, [-1, 1, np.prod(query_token_input_layer.shape.as_list()[1:])])
    #
    #     assert 4 == len(attention_input_layer_dict.keys()), 'attention attributes must be 4'
    #     # self attention
    #     with variable_scope.variable_scope('self'):
    #         # (?, T_k, C), (?, T_k, T_k)
    #         seq_vec, att_vec = multihead_attention(queries=seq_layer,
    #                                                keys=seq_layer,
    #                                                num_heads=len(attention_input_layer_dict.keys()),
    #                                                query_masks=key_masks,
    #                                                key_masks=key_masks,
    #                                                linear_projection=True,
    #                                                num_units=seq_layer.shape.as_list()[-1],
    #                                                num_output_units=seq_layer.shape.as_list()[-1],
    #                                                activation_fn='lrelu',
    #                                                atten_mode='ln',
    #                                                is_target_attention=False,
    #                                                first_n_att_weight_report=0,
    #                                                )
    #         att_vec = tf.concat(tf.split(att_vec, len(attention_input_layer_dict.keys()), axis=0), axis=2)
    #         seq_vec_shape, att_vec_shape = seq_vec.shape.as_list(), att_vec.shape.as_list()
    #         seq_vec = tf.reshape(seq_vec, [-1, seq_vec_shape[1] * seq_vec_shape[2]])
    #         att_vec = tf.reshape(att_vec, [-1, att_vec_shape[1] * att_vec_shape[2]])
    #         # projection network
    #         seq_vec = layers.fully_connected(
    #             seq_vec,
    #             seq_vec_shape[1],
    #             activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[0])
    #         )
    #         att_vec = layers.fully_connected(
    #             att_vec,
    #             att_vec_shape[1],
    #             activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[0])
    #         )
    #
    #         self_att_vec = tf.concat([seq_vec, att_vec], axis=-1)
    #         self_att_net = layers.batch_norm(self_att_vec, scale=True, is_training=self.co_model.is_training)

        # ******************************** delete cross network ********************************
        # # cross attention
        # with variable_scope.variable_scope('cross'):
        #     # (?, T_q, C), (?, T_q, T_k)
        #     seq_vec, att_vec = multihead_attention(queries=query_layer,
        #                                            keys=seq_layer,
        #                                            num_heads=len(attention_input_layer_dict.keys()),
        #                                            query_masks=None,
        #                                            key_masks=key_masks,
        #                                            linear_projection=True,
        #                                            num_units=query_layer.shape.as_list()[-1],
        #                                            num_output_units=seq_layer.shape.as_list()[-1],
        #                                            activation_fn='lrelu',
        #                                            atten_mode='ln',
        #                                            is_target_attention=True
        #                                            )
        #     att_vec = tf.concat(tf.split(att_vec, len(attention_input_layer_dict.keys()), axis=0), axis=2)
        #     seq_vec_shape, att_vec_shape = seq_vec.shape.as_list(), att_vec.shape.as_list()
        #     seq_vec = tf.reshape(seq_vec, [-1, seq_vec_shape[1] * seq_vec_shape[2]])
        #     att_vec = tf.reshape(att_vec, [-1, att_vec_shape[1] * att_vec_shape[2]])
        #     # projection network
        #     seq_vec = layers.fully_connected(
        #         seq_vec,
        #         seq_vec_shape[1],
        #         activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[0])
        #     )
        #     att_vec = layers.fully_connected(
        #         att_vec,
        #         att_vec_shape[1],
        #         activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[0])
        #     )

        #     cross_att_vec = tf.concat([seq_vec, att_vec], axis=-1)
        #     cross_att_net = layers.batch_norm(cross_att_vec, scale=True, is_training=self.co_model.is_training)

        # user attention
        # with variable_scope.variable_scope('user'):
        #     # (?, T_q, C), (?, T_q, T_k)
        #     seq_vec, att_vec = multihead_attention(queries=token_layer,
        #                                            keys=seq_layer,
        #                                            num_heads=len(attention_input_layer_dict.keys()),
        #                                            query_masks=tf.cast(tf.ones([tf.shape(token_layer)[0], 1]), tf.bool),
        #                                            key_masks=key_masks,
        #                                            linear_projection=True,
        #                                            num_units=token_layer.shape.as_list()[-1],
        #                                            num_output_units=seq_layer.shape.as_list()[-1],
        #                                            activation_fn='lrelu',
        #                                            atten_mode='ln',
        #                                            is_target_attention=False
        #                                            )
        #     att_vec = tf.concat(tf.split(att_vec, len(attention_input_layer_dict.keys()), axis=0), axis=2)
        #     seq_vec_shape, att_vec_shape = seq_vec.shape.as_list(), att_vec.shape.as_list()
        #     seq_vec = tf.reshape(seq_vec, [-1, seq_vec_shape[1] * seq_vec_shape[2]])
        #     att_vec = tf.reshape(att_vec, [-1, att_vec_shape[1] * att_vec_shape[2]])
        #     # projection network
        #     seq_vec = layers.fully_connected(
        #         seq_vec,
        #         seq_vec_shape[1],
        #         activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[0])
        #     )
        #     att_vec = layers.fully_connected(
        #         att_vec,
        #         att_vec_shape[1],
        #         activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[0])
        #     )
        #
        #     user_att_vec = tf.concat([seq_vec, att_vec], axis=-1)
        #     user_att_net = layers.batch_norm(user_att_vec, scale=True, is_training=self.co_model.is_training)
        #
        # # attention_output_layer = tf.concat([self_att_net, cross_att_net, user_att_net], axis=-1)
        # attention_output_layer = tf.concat([self_att_net, user_att_net], axis=-1)
        # return attention_output_layer

    # def _create_cross_network(self,
    #                       user_sparse_input_layer,
    #                       item_sparse_input_layer,
    #                       query_token_input_layer,
    #                       sequence_input_layer_dict,
    #                       dim=8,
    #                       units=256,
    #                       filters=32,
    #                       kernel_size=3):
    #     user_s_shape = user_sparse_input_layer.shape.as_list()[1] // dim
    #     user_sparse = tf.reshape(user_sparse_input_layer, [-1, user_s_shape, dim])
    #
    #     seq_sparse = []
    #     for name, layer in sequence_input_layer_dict.items():
    #         if layer.shape.as_list()[-1] > 1:
    #             seq_sparse.append(tf.reduce_mean(layer, axis=1, keep_dims=True))
    #     seq_sparse = tf.concat(seq_sparse, axis=1)
    #     user_sparse = tf.concat([user_sparse, seq_sparse], axis=1)
    #
    #     item_s_shape = item_sparse_input_layer.shape.as_list()[1] // dim
    #     item_sparse = tf.reshape(item_sparse_input_layer, [-1, item_s_shape, dim])
    #
    #     user_sparse = tf.transpose(user_sparse, [0, 2, 1])
    #     user_sparse = tf.layers.conv1d(user_sparse, filters=filters, kernel_size=kernel_size, padding='same')
    #     user_sparse = tf.transpose(user_sparse, [0, 2, 1])
    #
    #     item_sparse = tf.transpose(item_sparse, [0, 2, 1])
    #     item_sparse = tf.layers.conv1d(item_sparse, filters=filters, kernel_size=kernel_size, padding='same')
    #     item_sparse = tf.transpose(item_sparse, [0, 2, 1])
    #
    #     # user & item
    #     sparse_cross1 = tf.matmul(user_sparse, item_sparse, transpose_b=True)
    #     sparse_cross1 = tf.reshape(sparse_cross1, [-1, sparse_cross1.shape[1] * sparse_cross1.shape[2]])
    #
    #     # query & item
    #     query_token_sparse = tf.transpose(query_token_input_layer, [0, 2, 1])
    #     query_token_sparse = tf.layers.conv1d(query_token_sparse, filters=filters // 4, kernel_size=3, padding='same')
    #     query_token_sparse = tf.transpose(query_token_sparse, [0, 2, 1])
    #
    #     sparse_cross2 = tf.matmul(query_token_sparse, item_sparse, transpose_b=True)
    #     sparse_cross2 = tf.reshape(sparse_cross2, [-1, sparse_cross2.shape[1] * sparse_cross2.shape[2]])
    #
    #     # query & user
    #     sparse_cross3 = tf.matmul(query_token_sparse, user_sparse, transpose_b=True)
    #     sparse_cross3 = tf.reshape(sparse_cross3, [-1, sparse_cross3.shape[1] * sparse_cross3.shape[2]])
    #
    #     sparse_cross = tf.concat([sparse_cross1, sparse_cross2, sparse_cross3], axis=1)
    #
    #     with variable_scope.variable_scope('Sparse_Cross_Layer'):
    #         sparse_net = layers.fully_connected(
    #             sparse_cross,
    #             units,
    #             activation_fn=get_act_fn(self.co_model.dnn_hidden_units_act_op[0]),
    #         )
    #         sparse_net = layers.batch_norm(sparse_net, scale=True, is_training=self.co_model.is_training)
    #     return sparse_net


