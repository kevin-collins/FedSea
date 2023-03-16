# -*- coding: utf-8 -*-
from common.base.scheduler import BaseScheduler
from dataset.dataset_factory import DatasetFactory
from dataset.csv_dataset import get_dataset
from model.model_factory import ModelFactory
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np
import time
from utils.util import get_session

class LocalScheduler(BaseScheduler):

    def __init__(self, FLAGS, *args, **kwargs):
        self.FLAGS = FLAGS
        super(LocalScheduler, self).__init__(*args, **kwargs)

    def run(self):
        is_train = (self.FLAGS.mode == 'train')
        is_chief = (self.FLAGS.task_index == 0)

        train_batch_datas, test_batch_datas = get_dataset(self.FLAGS.train_files, self.FLAGS.test_files, self.FLAGS.batch_size, self.FLAGS.num_epochs if is_train else 1)

        print(self.FLAGS.model_name)
        model = ModelFactory.get_model(self.FLAGS.model_name, self.FLAGS)

        if is_train:
            model.build(train_batch_datas)
        else:
            model.build(test_batch_datas)

        self.stop_flag = tf.Variable(initial_value=0, name='stop_flag', trainable=False,
                                     collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        if self.FLAGS.same_init:
            self.avg_step = tf.Variable(initial_value=-1, name='avg_step', trainable=False,
                                        collections=[tf.GraphKeys.GLOBAL_VARIABLES])
        else:
            self.avg_step = tf.Variable(initial_value=0, name='avg_step', trainable=False,
                                        collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.avg_step_add = tf.assign_add(self.avg_step, 1, use_locking=True)

        self.index = tf.Variable(initial_value=range(self.FLAGS.country_nums), name='index', trainable=False,
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.now_round = 0
        self.start_time = time.time()

        self.gan_flag = tf.Variable(initial_value=0, name='gan_flag', trainable=False,
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES])
        self.gan_step_add = tf.assign_add(self.gan_flag, 1, use_locking=True)

        print("is_trian {}, is_chief {}".format(is_train,is_chief))

        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()]
        saver = tf.train.Saver()

        if is_train:
            with tf.Session() as sess:
                sess.graph.finalize()
                sess.run(init_op)

                coord = tf.train.Coordinator()
                thread = tf.train.start_queue_runners(sess, coord=coord, start=True)

                index_new = np.sort(
                    np.random.choice(len(self.FLAGS.countrys), self.FLAGS.country_nums,
                                     replace=False))

                self.index.load(index_new, sess)
                print("get new index", index_new)
                if is_train and self.FLAGS.agg and self.FLAGS.same_init and sess.run(self.avg_step) == -1:
                    print("start same init...")
                    global_vars = sess.run(tf.trainable_variables(scope='Global_SubModel'))

                    for i in range(len(self.FLAGS.countrys)):
                        t1 = time.time()
                        for variable, value in zip(tf.trainable_variables(
                                scope='{}_SubModel'.format(self.FLAGS.countrys[i])),
                                global_vars):
                            variable.load(value, sess)
                        print("{} is done,time is {}".format(self.FLAGS.countrys[i], time.time() - t1))

                    print("same init done!")
                    self.avg_step.load(0, sess)
                model.runner.before_run(sess)
                while self.now_round < self.FLAGS.round:
                    try:
                        model.runner.run(sess)
                        sess.run(self.avg_step_add)
                        now_avg_step = sess.run(self.avg_step)

                        # agg
                        if is_train and self.FLAGS.agg and now_avg_step > self.FLAGS.avg_freq:

                            print('start avg!')

                            sum_vars = None
                            var_nums = None

                            sum_weight = 0.0

                            # get sum vars
                            index_now = sess.run(self.index)
                            #

                            print("start calculate avg vars...")
                            for i in index_now:

                                local_vars = sess.run([v for v in tf.trainable_variables(
                                    scope='{}_SubModel'.format(self.FLAGS.countrys[i])) if
                                                           not any(key in v.name for key in self.FLAGS.avg_scope)])

                                print("last var_nums is {}, now {} var_nums is {}".format(var_nums, self.FLAGS.countrys[i],
                                                                                          len(local_vars)))

                                # make sure len is equal
                                if var_nums is None:
                                    var_nums = len(local_vars)
                                else:
                                    assert var_nums == len(local_vars)


                                local_weight = 1.
                                sum_weight += local_weight

                                if sum_vars is None:
                                    sum_vars = local_vars
                                    for i in range(len(sum_vars)):
                                        sum_vars[i] = local_vars[i] * local_weight
                                else:
                                    for i in range(len(sum_vars)):
                                        sum_vars[i] += local_vars[i] * local_weight

                            # get avg vars
                            global_vars = []
                            for var in sum_vars:
                                global_vars.append(var / float(sum_weight))
                            print("avg vars calculate done!")

                            for i in range(len(self.FLAGS.countrys)):
                                t1 = time.time()
                                for variable, value in zip([v for v in tf.trainable_variables(
                                        scope='{}_SubModel'.format(self.FLAGS.countrys[i])) if
                                                            not any(key in v.name for key in self.FLAGS.avg_scope)],
                                                           global_vars):
                                    variable.load(value, sess)
                                print("{} is done,time is {}".format(self.FLAGS.countrys[i], time.time() - t1))
                            print('avg done!')
                            saver.save(get_session(sess), self.FLAGS.ckpt_path+"/"+str(self.now_round)+"_model.ckpt")
                        # map process
                        if is_train and self.FLAGS.FedSFA:
                            if now_avg_step > self.FLAGS.avg_freq:
                                self.gan_flag.load(1, sess)
                                print("gan process start...")
                                for i in range(self.FLAGS.gan_map_step):
                                    model.runner.run(sess)
                                self.gan_flag.load(0, sess)
                                print("gan process is done")
                        # reset parameters
                        if is_train and now_avg_step > self.FLAGS.avg_freq:
                            # set round
                            print("round {}/{} end!  time:{}s".format(self.now_round, self.FLAGS.round,
                                                                      time.time() - self.start_time))
                            self.start_time = time.time()
                            self.now_round += 1

                            if is_train and self.now_round >= self.FLAGS.round:
                                # sign finish
                                self.stop_flag.load(1, sess)
                                print("round meets the requirement, chief out")
                                break

                            # load new index
                            index_new = np.sort(
                                np.random.choice(len(self.FLAGS.countrys), self.FLAGS.country_nums,
                                                 replace=False))

                            self.index.load(index_new, sess)
                            print("get new index", index_new)

                            # sign restart
                            self.avg_step.load(0, sess)

                            print("round {}/{} start...".format(self.now_round, self.FLAGS.round))
                    except tf.errors.OutOfRangeError:
                        break
                model.runner.after_run(sess)
                coord.request_stop()
                coord.join(thread)
                print('Finish.')
        else:
            with tf.Session() as sess:
                sess.graph.finalize()
                sess.run(init_op)
                saver.restore(sess, self.FLAGS.ckpt_path)

                coord = tf.train.Coordinator()
                thread = tf.train.start_queue_runners(sess, coord=coord, start=True)
                model.runner.before_run(sess)
                while True:
                    try:
                        model.runner.run(sess)
                    except tf.errors.OutOfRangeError:
                        print("run of of data")
                        break
                coord.request_stop()
                coord.join(thread)
                aucs = []
                for i in range(len(self.FLAGS.countrys)):
                    auc = roc_auc_score(model.runner.labels[i],model.runner.preds[i])
                    aucs.append(auc)
                    print("{} auc is {}".format(self.FLAGS.countrys[i],auc))
                print("avg auc is {}".format(np.average(aucs)))