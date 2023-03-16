# -*- coding: utf-8 -*-
import time
from abc import ABCMeta, abstractmethod
from common.base.mode import ModeKeys
from common.base.util import logger
import tensorflow as tf
import numpy as np
import os
import sys
import datetime
from utils.util import mk_mmd_loss,caculate_model_distance
import scipy.stats as stats
from utils.util import get_mdl_params,set_client_from_params

class BaseRunner(object):
    __metaclass__ = ABCMeta

    def __init__(self, FLAGS, *args, **kwargs):
        self.FLAGS = FLAGS
        self._train_ops = []
        self._inference_ops = []
        self._evaluate_ops = []
        self._log_ops = [[], []]

        self.local_train_step = 0
        self.local_test_step = 0
        self.local_predict_step = 0
        self.local_last_step = 0
        self.during_step = 1000 if self.FLAGS.mode != ModeKeys.LOCAL else 1
        self.odps_writer = None


    def add_train_ops(self, ops):
        self._train_ops.extend(ops)

    def add_inference_ops(self, ops):
        self._inference_ops.extend(ops)

    def add_evaluate_ops(self, ops):
        self._evaluate_ops.extend(ops)

    def add_log_ops(self, names, ops):
        assert len(names) == len(ops), 'name number must match with ops'
        self._log_ops[0].extend(names)
        self._log_ops[1].extend(ops)

    def before_run(self, sess, *args, **kwargs):
        pass

    def run(self, sess, *args, **kwargs):
        if self.FLAGS.mode == ModeKeys.LOCAL:
            return self._train(sess)
        elif self.FLAGS.mode == ModeKeys.PREDICT:
            return self._inference(sess)
        elif self.FLAGS.mode == ModeKeys.TRAIN:
                if self.FLAGS.task_index == 0:  # worker 0 for validation
                    return self._evaluate(sess)
                else:  # other workers for training
                    return self._train(sess)
        elif self.FLAGS.mode == ModeKeys.EVAL:
            return self._evaluate(sess)
        else:
            raise Exception('Unsupported Mode:{}'.format(self.FLAGS.mode))

    def after_run(self, sess, *args, **kwargs):
        pass

    def _train(self, sess, *args, **kwargs):
        feed_dict = {'training:0': True}
        result = sess.run(self._log_ops[1] + self._train_ops, feed_dict=feed_dict)
        if self.local_train_step % self.during_step == 0:
            print('Train:{}'.format(', '.join(['{}={:.7g}'.format(k, v) for k, v in zip(self._log_ops[0], result)])))
            sys.stdout.flush()
        self.local_train_step += 1
        return False

    def _evaluate(self, sess, *args, **kwargs):
        feed_dict = {'training:0': False}
        result = sess.run(self._log_ops[1] + self._evaluate_ops + [tf.train.get_global_step()], feed_dict=feed_dict)
        if self.local_test_step % self.during_step == 0:
            print('Validation:{}'.format(', '.join(['{}={:.7g}'.format(k, v) for k, v in zip(self._log_ops[0], result)])))
            sys.stdout.flush()
            if result[-1] > self.local_last_step:
                self.local_last_step = result[-1]
            else:
                print('global_step not updating during {} test steps'.format(self.during_step))
                sys.stdout.flush()
                version = tf.VERSION.split('.')
                if int(version[0]) == 1 and int(version[1]) <= 8:
                    sess.request_stop(notify_all=True)
                return True
        self.local_test_step += 1

    def _inference(self, sess, *args, **kwargs):
        feed_dict = {'training:0': False}
        result = sess.run(self._inference_ops, feed_dict=feed_dict)
        if self.FLAGS.outputs is not None and self.odps_writer is None:
            self.odps_writer = tf.python_io.TableWriter(self.FLAGS.outputs, slice_id=self.FLAGS.task_index)
        values = []
        for i in range(len(result[0])):
            lst = []
            for j in range(len(result)):
                lst.append(','.join([str(t) for t in np.ravel(result[j][i])]))
            values.append([';'.join(lst)])
        self.odps_writer.write(values, indices=[0])
        if self.local_predict_step % self.during_step == 0:
            print('Predict step {}: {}'.format(self.local_predict_step, values[0]))
            sys.stdout.flush()
        self.local_predict_step += 1
        return False


class Runner(BaseRunner):

    def __init__(self, FLAGS, *args, **kwargs):
        super(Runner, self).__init__(FLAGS, *args, **kwargs)
        self.save_summaries_steps = FLAGS.save_summaries_steps
        self.local_summary_step = 0
        self.summary_op = None
        self.summary_writer = None

        if FLAGS.task_index == 0:
            self.summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.checkpointDir, 'Test'))
        elif FLAGS.task_index == 1:
            self.summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.checkpointDir, 'Train'))

    def set_summary_op(self, op):
        self.summary_op = op

    def write_summary(self, sess, summary_op, global_step):
        if self.summary_writer is None:
            return
        if global_step // self.save_summaries_steps > self.local_summary_step:
            if self.local_summary_step == 0:
                self.summary_writer.add_graph(sess.graph)
            self.summary_writer.add_summary(summary_op, global_step)
            self.summary_writer.flush()
            self.local_summary_step = global_step // self.save_summaries_steps

    def _train(self, sess, *args, **kwargs):
        feed_dict = {'training:0': True}
        result = sess.run(self._log_ops[1] + self._train_ops + [self.summary_op, tf.train.get_global_step()], feed_dict=feed_dict)
        self.write_summary(sess, result[-2], result[-1])
        if self.local_train_step % self.during_step == 0:
            print('Train:{}'.format(', '.join(['{}={:.7g}'.format(k, v) for k, v in zip(self._log_ops[0], result)])))
            sys.stdout.flush()
        self.local_train_step += 1
        return False

    def _evaluate(self, sess, *args, **kwargs):
        feed_dict = {'training:0': False}
        result = sess.run(self._log_ops[1] + self._evaluate_ops + [self.summary_op, tf.train.get_global_step()], feed_dict=feed_dict)
        self.write_summary(sess, result[-2], result[-1])
        if self.local_test_step % self.during_step == 0:
            print('Validation:{}'.format(', '.join(['{}={:.7g}'.format(k, v) for k, v in zip(self._log_ops[0], result)])))
            sys.stdout.flush()
            if result[-1] > self.local_last_step:
                self.local_last_step = result[-1]
            else:
                print('global_step not updating during {} test steps'.format(self.during_step))
                sys.stdout.flush()
                version = tf.VERSION.split('.')
                if int(version[0]) == 1 and int(version[1]) <= 8:
                    sess.request_stop(notify_all=True)
                return True
        self.local_test_step += 1
        return False



class CotrainRunner(object):
    __metaclass__ = ABCMeta

    def __init__(self, FLAGS, *args, **kwargs):
        self.FLAGS = FLAGS
        self._train_ops = []
        self._train_stage2_ops = []
        self._inference_ops = []
        self._evaluate_ops = []
        self._log_ops = [[], []]
        self._dis_ops = []
        self._calsim_ops = []
        self._analysis_ops = []

        self.local_train_step = 0
        self.local_test_step = 0
        self.local_predict_step = 0
        self.local_last_step = 0
        self.during_step = 100 if self.FLAGS.mode != ModeKeys.LOCAL else 1
        self.odps_writer = None
        self.alternate_ratio = self.FLAGS.alternate_ratio
        self.times = 0

        self.global_vars = None
        self.index = [i for i in range(len(self.FLAGS.countrys))]
        self.cal_sim = None
        self.sample_nums = [0 for _ in range(len(self.FLAGS.countrys))]
        self.batch_sample_nums = [0 for _ in range(len(self.FLAGS.countrys))]

        self.global_step = 0

        self.p = 0.0

        self.labels = [[] for _ in range(len(self.FLAGS.countrys))]
        self.preds = [[] for _ in range(len(self.FLAGS.countrys))]

    def add_train_ops(self, ops):
        self._train_ops.extend(ops)

    def add_train_stage2_ops(self, ops):
        self._train_stage2_ops.extend(ops)

    def add_inference_ops(self, ops):
        self._inference_ops.extend(ops)

    def add_evaluate_ops(self, ops):
        self._evaluate_ops.extend(ops)

    def add_log_ops(self, names, ops):
        assert len(names) == len(ops), 'name number must match with ops'
        self._log_ops[0].extend(names)
        self._log_ops[1].extend(ops)

    def add_dis_ops(self, ops):
        self._dis_ops.extend(ops)

    def add_calsim_ops(self, ops):
        self._calsim_ops.extend(ops)

    def add_analysis_ops(self,ops):
        self._analysis_ops.extend(ops)

    def before_run(self, sess, *args, **kwargs):
        pass

    def run(self, sess, *args, **kwargs):
        if self.FLAGS.mode == ModeKeys.LOCAL:
            return self._train(sess)
        elif self.FLAGS.mode == ModeKeys.PREDICT or self.FLAGS.mode == ModeKeys.GLOBAL:
            return self._inference(sess)
        elif self.FLAGS.mode == ModeKeys.TRAIN:
                if "distributed" in self.FLAGS.scheduler_name and self.FLAGS.task_index == 0:
                    return self._evaluate(sess)
                if self.FLAGS.FedSFA and sess.run(tf.global_variables(scope='gan_flag'))[0]>=1:
                    return self._gan(sess)
                else:
                    return self._train(sess)
        elif self.FLAGS.mode == ModeKeys.EVAL:
            return self._evaluate(sess)
        else:
            raise Exception('Unsupported Mode:{}'.format(self.FLAGS.mode))

    def after_run(self, sess, *args, **kwargs):
        print("run times is ",self.times)
        pass

    def _train(self, sess, *args, **kwargs):
        feed_dict = {'training:0': True}

        if self.FLAGS.agg:
            index = sess.run(tf.global_variables(scope='index'))[0]
        else:
            index = np.arange(len(self.FLAGS.countrys))

        self.index = index


        result = sess.run([ self._log_ops[1][i]+self._train_ops[i] for i in self.index ]+[self._train_ops[-1]],
                          feed_dict=feed_dict)

        self.global_step = int(result[-1])

        self.p = min(self.local_train_step / 200000.,1.)
        # self.p = 2./(1.+np.exp(-10*self.local_train_step)) - 1.

        # log
        if self.local_train_step % self.during_step == 0 or self.local_train_step < 10:
            print(datetime.datetime.now())
            # for each element in index
            print("index is",self.index)
            for i in range(len(self.index)):
                print(self.FLAGS.countrys[self.index[i]] + ' Train:{}'.format(
                    ', '.join(['{}={}'.format(k, v) for k, v in zip(self._log_ops[0][self.index[i]], result[i])])))
            print("##########################")

        self.local_train_step += 1

        return False

    def _gan(self, sess, *args, **kwargs):
        feed_dict = {'training:0': True}

        dis_result = sess.run(self._dis_ops, feed_dict=feed_dict)
        print("dis_loss:{:.4f} dis_auc:{:4f}".format(dis_result[1], dis_result[2]))

        return False

    def _evaluate(self, sess, *args, **kwargs):
        feed_dict = {'training:0': False}

        index = np.arange(len(self.FLAGS.countrys))
        result = sess.run([[self._log_ops[1][i]+self._evaluate_ops[i] for i in index]]+[tf.train.get_global_step()],
                          feed_dict=feed_dict)
        auc = 0.
        if self.local_test_step % self.during_step == 0:
            print(datetime.datetime.now())
            print("test_step",self.local_test_step)
            for i in range(len(self.FLAGS.countrys)):
                print(self.FLAGS.countrys[index[i]] + ' Validation:{}'.format(
                    ', '.join(['{}={}'.format(k, v) for k, v in zip(self._log_ops[0][index[i]], result[0][i])])))
                auc += result[0][i][2]

            auc = auc / len(self.FLAGS.countrys)
            print("avg auc: {}".format(auc))
            print("#############")


        if result[-1] > self.local_last_step:
            self.local_last_step = result[-1]
        else:
            if result[-1]>1000 and self.local_test_step % (50 * self.during_step) == 0:
                print("global_step {} local_last_step {}".format(result[-1],self.local_last_step))
                print('global_step not updating during {} test steps'.format(50 * self.during_step))
                version = tf.VERSION.split('.')
                if int(version[0]) == 1 and int(version[1]) <= 8:
                    sess.request_stop(notify_all=True)
                return True

        self.local_test_step += 1
        self.times += 1

        return False

    def _inference(self, sess, *args, **kwargs):
        feed_dict = {'training:0': False}

        result = sess.run([self._inference_ops[i] for i in range(len(self.FLAGS.countrys))], feed_dict=feed_dict)

        # for each country
        for i in range(len(result)):
            # for each data
            for j in range(len(result[i][0])):
                # for each feature
                for k in range(len(result[i])):
                    if k==1:
                        self.labels[i].extend(result[i][k][j])
                    elif k==3:
                        self.preds[i].extend(result[i][k][j])

        if self.local_predict_step % self.during_step == 0:
            print('Predict step {}: {}'.format(self.local_predict_step, self.preds))
        self.local_predict_step += 1
        return False

class ForceRunner(BaseRunner):

    def __init__(self, FLAGS, *args, **kwargs):
        super(ForceRunner, self).__init__(FLAGS, *args, **kwargs)
        self._stop_flag = tf.get_variable(name='stop_flag',
                                          dtype=tf.bool,
                                          trainable=False,
                                          collections=[tf.GraphKeys.GLOBAL_VARIABLES],
                                          initializer=False)
        self._update_flag = tf.assign(self._stop_flag, True)

    def _train(self, sess, *args, **kwargs):
        feed_dict = {'training:0': True}
        result = sess.run(self._log_ops[1] + self._train_ops + [self._stop_flag], feed_dict=feed_dict)
        if self.local_train_step % self.during_step == 0:
            print('Train:{}'.format(', '.join(['{}={:.7g}'.format(k, v) for k, v in zip(self._log_ops[0], result)])))
            sys.stdout.flush()
        self.local_train_step += 1
        return result[-1]

    def _evaluate(self, sess, *args, **kwargs):
        feed_dict = {'training:0': False}
        result = sess.run(self._log_ops[1] + self._evaluate_ops + [tf.train.get_global_step()], feed_dict=feed_dict)
        if self.local_test_step % self.during_step == 0:
            print('Validation:{}'.format(', '.join(['{}={:.7g}'.format(k, v) for k, v in zip(self._log_ops[0], result)])))
            sys.stdout.flush()
            if result[-1] > self.local_last_step:
                self.local_last_step = result[-1]
            else:
                print('global_step not updating during {} test steps'.format(self.during_step))
                sys.stdout.flush()
                sess.run(self._update_flag)
                print('update stop flag to be true')
                version = tf.VERSION.split('.')
                if int(version[0]) == 1 and int(version[1]) <= 8:
                    sess.request_stop(notify_all=True)
                return True
        self.local_test_step += 1
        return False

    def _inference(self, sess, *args, **kwargs):
        feed_dict = {'training:0': False}
        result = sess.run(self._inference_ops, feed_dict=feed_dict)
        if self.FLAGS.outputs is not None and self.odps_writer is None:
            self.odps_writer = tf.python_io.TableWriter(self.FLAGS.outputs, slice_id=self.FLAGS.task_index)
        values = []
        for i in range(len(result[0])):
            lst = []
            for j in range(len(result)):
                lst.append(','.join([str(t) for t in np.ravel(result[j][i])]))
            values.append([';'.join(lst)])
        self.odps_writer.write(values, indices=[0])
        if self.local_predict_step % self.during_step == 0:
            print('Predict step {}: {}'.format(self.local_predict_step, values[0]))
            sys.stdout.flush()
        self.local_predict_step += 1
        return False

class ADSRunner(object):
    def __init__(self, FLAGS, *args, **kwargs):
        self.FLAGS = FLAGS
        self.mode = FLAGS.mode
        self._train_ops = []
        self._inference_ops = []
        self._evaluate_ops = []
        self._log_ops = [[], []]

        self.local_step = 0
        self.local_last_step = 0
        self.train_step = FLAGS.train_step if self.mode != ModeKeys.LOCAL else 1
        self.test_step = FLAGS.test_step if self.mode != ModeKeys.LOCAL else 1
        self.odps_writer = None

    def add_train_ops(self, ops):
        self._train_ops.extend(ops)

    def add_inference_ops(self, ops):
        self._inference_ops.extend(ops)

    def add_evaluate_ops(self, ops):
        self._evaluate_ops.extend(ops)

    def add_log_ops(self, names, ops):
        assert len(names) == len(ops), 'name number must match with ops'
        self._log_ops[0].extend(names)
        self._log_ops[1].extend(ops)

    def before_run(self, sess, *args, **kwargs):
        pass

    def run(self, sess, *args, **kwargs):
        if self.mode == ModeKeys.LOCAL:
            return self._train(sess)
        elif self.mode == ModeKeys.PREDICT:
            return self._inference(sess)
        elif self.mode == ModeKeys.TRAIN:
            if self.FLAGS.task_index == 0:  # worker 0 for validation
                return self._evaluate(sess)
            else:  # other workers for training
                return self._train(sess)
        elif self.mode == ModeKeys.EVAL:
            return self._evaluate(sess)
        else:
            raise Exception('Unsupported Mode:{}'.format(self.mode))

    def after_run(self, sess, *args, **kwargs):
        pass

    def display_log(self, prefix, ops, result):
        logger('{}: local_step={}, {}'.format(prefix, self.local_step, ', '.join(['{}={:.7g}'.format(k, v) for k, v in zip(ops, result)])))
        sys.stdout.flush()

    def _train(self, sess, *args, **kwargs):
        if self.local_step == 0:
            logger('start training...')
        feed_dict = {'training:0': True}
        result = sess.run(self._log_ops[1] + self._train_ops, feed_dict=feed_dict)
        self.local_step += 1

        if self.local_step % self.train_step == 0:
            self.display_log('Train', self._log_ops[0], result)
        return False

    def _evaluate(self, sess, *args, **kwargs):
        if self.local_step == 0:
            logger('start evaluating...')
        feed_dict = {'training:0': False}
        result = sess.run(self._log_ops[1] + self._evaluate_ops + [tf.train.get_global_step()], feed_dict=feed_dict)
        self.local_step += 1

        if self.local_step % self.test_step == 0:
            self.display_log('Validation', self._log_ops[0], result)

            if result[-1] <= 0:
                logger('No eval data available, please wait or kill this job...')
            elif result[-1] > self.local_last_step:
                self.local_last_step = result[-1]
            else:
                logger('global_step not updating during {} test steps'.format(self.test_step))
                sys.stdout.flush()
                version = tf.VERSION.split('.')
                if int(version[0]) == 1 and int(version[1]) <= 8:
                    sess.request_stop(notify_all=True)
                return True
        return False

    def _inference(self, sess, *args, **kwargs):
        if self.local_step == 0:
            logger('start predicting...')
        feed_dict = {'training:0': False}
        result = sess.run(self._inference_ops, feed_dict=feed_dict)
        if self.FLAGS.outputs is not None and self.odps_writer is None:
            self.odps_writer = tf.python_io.TableWriter(self.FLAGS.outputs, slice_id=self.FLAGS.task_index)
        values = []
        for i in range(len(result[0])):
            lst = []
            for j in range(len(result)):
                lst.append(','.join([str(t) for t in np.ravel(result[j][i])]))
            values.append([';'.join(lst)])
        self.odps_writer.write(values, indices=[0])
        self.local_step += 1

        if self.local_step % self.train_step == 0:
            logger('Predict step {}: {}'.format(self.local_step, values[0]))
            sys.stdout.flush()
        return False
