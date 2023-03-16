# -*- coding: utf-8 -*-
from __future__ import print_function

from common.base.model import BaseModel
from common.base.runner import CotrainRunner
from tensorflow.contrib import layers
from common.base.mode import ModeKeys
from model.ae_search_ha3_model.sub_model_fed import SubModel
from utils.config import parse_model_conf
from utils.util import random_shuffle
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from utils.util import get_act_fn

class CotrainModel(BaseModel):

    def __init__(self, FLAGS, *args, **kwargs):
        super(CotrainModel, self).__init__(FLAGS, *args, **kwargs)
        self.FLAGS = FLAGS
        self.runner = CotrainRunner(FLAGS, *args, **kwargs)

        # job config
        self.ps_num = len(self.FLAGS.ps_hosts.split(','))

        # network hyper parameters
        self.dnn_l2_reg = FLAGS.dnn_l2_reg
        self.learning_rate = FLAGS.learning_rate
        self.embedding_partition_size = FLAGS.embedding_partition_size
        self.need_dropout = FLAGS.need_dropout
        self.dropout_rate = FLAGS.dropout_rate
        self.user_output_size = FLAGS.user_output_size
        self.item_output_size = FLAGS.item_output_size
        self.dnn_hidden_units = FLAGS.dnn_hidden_units
        self.dnn_hidden_units_act_op = FLAGS.dnn_hidden_units_act_op
        self.map_hidden_units = FLAGS.map_hidden_units

        self.local_dnn_hidden_units = FLAGS.local_dnn_hidden_units
        self.local_dnn_hidden_units_act_op = FLAGS.local_dnn_hidden_units_act_op
        self.combine_hidden_units = FLAGS.combine_hidden_units
        ######################################################################################################################
        #                                                                                                                    #
        #                                               Feature Group                                                        #
        #                                                                                                                    #
        ######################################################################################################################
        # get name list
        parse_model_conf(FLAGS)
        column_conf = FLAGS.mc_conf['input_columns']
        self.column_conf = column_conf

        self.user_sparse_features = column_conf['user_sparse']
        self.user_dense_features = column_conf['user_dense']
        self.user_behavior_features = column_conf['user_behavior']

        self.item_sparse_features = column_conf['item_sparse']
        self.item_dense_features = column_conf['item_dense']
        self.item_behavior_features = column_conf['item_behavior']

        self.query_sparse_features = column_conf['query_sparse']

        ######################################################################################################################
        #                                                                                                                    #
        #                                               Graph Node                                                           #
        #                                                                                                                    #
        ######################################################################################################################
        self.clients = []
        self.sample_nums = []
        for country in self.FLAGS.countrys:
            self.clients.append(SubModel(self, country))
            self.sample_nums.append(None)

        self.global_model = SubModel(self, 'Global')

        self.reg_loss = None
        self.loss = None
        self.global_step = None
        self.is_training = None

        self.mask = None
        self.out = None

        self.logits = None
        self.labels = None

        self.loss = None

    def build(self, batch_data, *args, **kwargs):
        self._build_preliminary()
        self._build_inputs(batch_data)
        self._build_model()
        self._build_loss()
        self._build_optimizer()
        self._build_rtp()
        self._build_summary()
        self._build_runner()

    def _build_preliminary(self):
        # tf.get_default_graph().set_shape_optimize(False)
        try:
            training = tf.get_default_graph().get_tensor_by_name('training:0')
        except KeyError:
            training = tf.placeholder_with_default(False, shape=(), name='training')
        self.is_training = training
        self.global_step = tf.Variable(initial_value=0, name='global_step', trainable=False,
                                       collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)


    def _parse_generated_fg(self, batch_data):
        with tf.name_scope('Input_Pipeline'):
            import rtp_fg
            fg_features = rtp_fg.parse_genreated_fg(self.FLAGS.fg_conf, batch_data[1])
            fg_features['label'] = batch_data[3]
            fg_features['id'] = batch_data[0]

            if getattr(self.FLAGS, 'enable_stage', False):
                print("Enable stage")
                with tf.device('CPU:0'):
                    fg_features = tf.staged(fg_features, capacity=2, num_threads=1, closed_exception_types=(tf.errors.OutOfRangeError,))
            else:
                print("Disable stage")
            return fg_features


    def _build_inputs(self, batch_data_countrys):
        for idx,batch_data in enumerate(batch_data_countrys):
            # ctr_batch = batch_data.get('ctr_batch_data', None)
            ctr_fg_features = self._parse_generated_fg(batch_data)

            ctr_label = tf.to_float(tf.greater(tf.string_to_number(ctr_fg_features['label'], out_type=tf.float32), 0.5))

            ctr_label = tf.reshape(ctr_label, [-1, 1])
            print(len(self.clients), idx)
            self.clients[idx].build_inputs(ctr_fg_features, ctr_label)

            # if (self.FLAGS.moe or self.FLAGS.fedprox)and idx==0:

            if idx==0:
                self.global_model.build_inputs(ctr_fg_features, ctr_label)

            # for infer cal auc
            self.ctr_sample_id = tf.reshape(ctr_fg_features['id'], [-1, 1])
            self.ctr_real_label = tf.reshape(ctr_fg_features['label'], [-1, 1])

            if self.sample_nums[idx] == None:
                self.sample_nums[idx] = tf.shape(ctr_label)[0]
            else:
                self.sample_nums[idx] += tf.shape(ctr_label)[0]

            self.clients[idx].ctr_sample_id = tf.reshape(ctr_fg_features['id'], [-1, 1])
            self.clients[idx].ctr_real_label = tf.reshape(ctr_fg_features['label'], [-1, 1])



    def _build_model(self):

        if self.FLAGS.mode == ModeKeys.GLOBAL:
            self.global_model.build_model()
            return

        self.global_model.build_model()

        for c in self.clients:
            c.build_model()

        # train dis
        if self.FLAGS.FedSFA:
            self.dis_data = None
            self.dis_label = None
            self.pred_label = None
            for i in range(len(self.FLAGS.countrys)):
                if self.dis_data is None:
                    self.dis_data = self.clients[i].ori_feature
                    self.dis_label = self.clients[i].dis_label
                    self.pred_label = self.clients[i].labels
                else:
                    self.dis_data = tf.concat([self.dis_data, self.clients[i].ori_feature], axis=0)
                    self.dis_label = tf.concat([self.dis_label, self.clients[i].dis_label], axis=0)
                    self.pred_label = tf.concat([self.pred_label, self.clients[i].labels], axis=0)

            # shuffle sample one batch from all client map data
            self.dis_data, self.dis_label, self.pred_label = random_shuffle(self.dis_data, self.dis_label, self.pred_label)

            self.dis_data = self.dis_data
            self.dis_label = self.dis_label
            self.pred_label = self.pred_label

            dis_data = self.dis_data
            with variable_scope.variable_scope('Map_Net', reuse=tf.AUTO_REUSE):
                for layer_id, num_hidden_units in enumerate([64, 128, len(self.FLAGS.countrys)]):
                    with variable_scope.variable_scope('Hidden_{}'.format(layer_id)):
                        dis_data = layers.fully_connected(dis_data,
                                                                     num_hidden_units,
                                                                     activation_fn=get_act_fn(
                                                                         self.dnn_hidden_units_act_op[
                                                                             layer_id]),
                                                                     )
                self.dis_logits = dis_data



        for v in tf.trainable_variables():
            if 'conv1d' in v.name:
                tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, v)


    def _build_loss(self):
        with tf.name_scope('Loss'):
            if self.FLAGS.mode == ModeKeys.GLOBAL:
                self.global_model.build_loss()
                return
            for c in self.clients:
                c.build_loss()

        if self.FLAGS.FedSFA:
            self.dis_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.dis_label, logits=self.dis_logits)
            )
            pred = tf.cast(tf.argmax(self.dis_logits, axis=1), tf.float32)
            dis_label = tf.cast(tf.argmax(self.dis_label, axis=1), tf.float32)
            self.acc = 1 - tf.cast(tf.count_nonzero(dis_label - pred), tf.float32) / (512. * 4.)

    def _build_optimizer(self):
        if self.FLAGS.mode == ModeKeys.GLOBAL:
            return

        for c in self.clients:
            c.build_optimizer()

        if self.FLAGS.FedSFA:
            dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Map_Net')
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)  # for TF1120
            self.dis_op = optimizer.minimize(self.dis_loss,var_list=dis_vars)

    def _build_rtp(self):
        with tf.name_scope('Mark_Output'):
            pass

    def _build_summary(self):
        if self.FLAGS.mode == ModeKeys.GLOBAL:
            self.global_model.build_summary()
            return

        for c in self.clients:
            c.build_summary()

    def _build_runner(self):
        if self.FLAGS.mode == ModeKeys.GLOBAL:
            self.runner.add_evaluate_ops([[self.global_model.auc_update, self.global_model.predicts, self.global_model.loss]])
            self.runner.add_log_ops([['global_step', 'local_ctr_step', 'ctr_auc', 'ctr_loss', 'label_mean']],
                                    [[self.global_step, self.global_model.train_step, self.global_model.auc,
                                      self.global_model.loss, self.global_model.label_mean]])
            self.runner.add_inference_ops([[self.global_model.ctr_sample_id, self.global_model.ctr_real_label,
                                            self.global_model.labels, self.global_model.predicts]])
        else:
            if self.FLAGS.FedSFA:
                self.runner.add_dis_ops([self.dis_op, self.dis_loss,self.acc,self.dis_data,self.dis_label,self.dis_logits])

            for i in range(len(self.FLAGS.countrys)):
                c = self.clients[i]
                self.runner.add_train_ops([[c.train_op, c.auc_update,c.input_dict, self.sample_nums[i]]])

                self.runner.add_evaluate_ops([[c.auc_update, c.predicts, c.loss]])
                if self.FLAGS.FedSFA:
                    self.runner.add_log_ops(
                        [['global_step', 'local_ctr_step', 'ctr_auc', 'ctr_loss', 'dis_loss', 'label_mean','mask_global']],
                        [[self.global_step, c.train_step, c.auc, c.predict_loss, c.dis_loss, c.label_mean,c.mask_global]])

                else:
                    self.runner.add_log_ops([['global_step', 'local_ctr_step', 'ctr_auc', 'ctr_loss','label_mean','labels']],
                                        [[self.global_step, c.train_step, c.auc, c.loss,c.label_mean,c.labels]])
                self.runner.add_inference_ops([[c.ctr_sample_id, c.ctr_real_label, c.labels, c.predicts]])
        self.runner.add_train_ops([self.global_step_add])


    def get_scaffold(self):
        max_to_keep = getattr(self.FLAGS, 'max_to_keep', 5)
        if self.FLAGS.task_index != 0:
            print('non chief worker with default scaffold.')
            # mysaver = tf.train.Saver(max_to_keep=max_to_keep, sharded=True, builder=pai.utils.PartialRestoreSaverBuilder())
            # tf.add_to_collection(tf.GraphKeys.SAVERS, mysaver)
            # return tf.train.Scaffold(saver=mysaver)
            return tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=max_to_keep, sharded=True))

        load_checkpoint_dir = getattr(self.FLAGS, 'load_checkpoint_dir', None)
        print( 'load_checkpoint_dir is',load_checkpoint_dir)
        if not load_checkpoint_dir:
            print('chief worker with default scaffold.')
            # mysaver = tf.train.Saver(max_to_keep=max_to_keep, sharded=True, builder=pai.utils.PartialRestoreSaverBuilder())
            # tf.add_to_collection(tf.GraphKeys.SAVERS, mysaver)
            # return tf.train.Scaffold(saver=mysaver)
            return tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=max_to_keep, sharded=True))
        else:
            print('load checkpoint from:{}'.format(load_checkpoint_dir))
            ckpt = tf.train.get_checkpoint_state(load_checkpoint_dir)
            if ckpt is None:
                print('no checkpoint at:{}, using default scaffold.'.format(load_checkpoint_dir))
                return tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=max_to_keep, sharded=True))
            else:
                # var to restore in graph
                load_embed_only = getattr(self.FLAGS, 'load_embed_only', True)
                print("load_embed_only", load_embed_only)
                if load_embed_only:
                    variables_to_restore = [v for v in tf.trainable_variables() if 'embedding' in v.name.lower()]
                else:
                    variables_to_restore = tf.trainable_variables()
                # variables_to_restore = tf.trainable_variables()
                # merge different parts of the same variable
                var_dict = {}
                for v in variables_to_restore:
                    name = v.name.split(':')[0].split('/')
                    if name[-1].startswith('part_'):
                        name.pop(-1)
                    name = '/'.join(name)
                    if name not in var_dict:
                        var_dict[name] = [v]
                    else:
                        var_dict[name].append(v)
                print("load_var_list:")
                print(var_dict.keys())
                init_assign_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(ckpt.model_checkpoint_path,
                                                                                             var_dict,
                                                                                             ignore_missing_vars=True)

                def init_fn(_scaffold, sess):
                    sess.run(init_assign_op, init_feed_dict)
                scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=max_to_keep, sharded=True), init_fn=init_fn)
                return scaffold
