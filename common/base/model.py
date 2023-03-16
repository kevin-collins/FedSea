from abc import ABCMeta, abstractmethod
from common.base.runner import BaseRunner, ForceRunner
from tensorflow.python.ops import variables
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


class BaseModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, FLAGS, *args, **kwargs):
        self.FLAGS = FLAGS
        self.runner = BaseRunner(FLAGS)

    @abstractmethod
    def build(self, *args, **kwargs):
        raise NotImplementedError('Calling an abstract method.')

    def get_scaffold(self):
        max_to_keep = getattr(self.FLAGS, 'max_to_keep', 5)
        load_checkpoint_dir = getattr(self.FLAGS, 'load_checkpoint_dir', None)
        load_embedding_only = getattr(self.FLAGS, 'load_embedding_only', None)
        if not load_checkpoint_dir:
            return tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=max_to_keep, sharded=True))
        else:
            print('load checkpoint from:{}'.format(load_checkpoint_dir))
            ckpt = tf.train.get_checkpoint_state(load_checkpoint_dir)
            if ckpt is None:
                print('no checkpoint at:{}'.format(load_checkpoint_dir))
                return tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=max_to_keep, sharded=True))
            else:
                reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
                ckpt_var_map = reader.get_variable_to_shape_map()
                model_vars = [var for var in tf.contrib.framework.get_variables_to_restore() if 'Adam' not in var.name]
                variables_to_restore = []

                for var_name, var_shape in ckpt_var_map.items():
                    if 'Adam' not in var_name:
                        for each in model_vars:
                            if var_name in each.name and var_shape == each.get_shape().as_list():
                                print(each.name)
                                print(each.get_shape().as_list(), var_name)
                                variables_to_restore.append(each)

                if load_embedding_only:
                    print('load embedding only')
                    variables_to_restore = [x for x in variables_to_restore if 'shared_embedding' in x.name]

                print('vars to restore:')
                print('\n'.join([x.name for x in variables_to_restore]))
                init_assign_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(ckpt.model_checkpoint_path,
                                                                                             variables_to_restore,
                                                                                             ignore_missing_vars=True)

                def init_fn(scaffold, sess):
                    sess.run(init_assign_op, init_feed_dict)
                scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=max_to_keep, sharded=True), init_fn=init_fn)
                return scaffold

    def get_hooks(self):
        hooks = []
        loss = getattr(self, 'loss', None)
        if loss is not None:
            nan_hook = tf.train.NanTensorHook(loss)
            hooks.append(nan_hook)
        return hooks

    @staticmethod
    def get_chief_only_hooks():
        return None

    @staticmethod
    def get_schedule_config():
        conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        conf.gpu_options.allow_growth = True
        conf.inter_op_parallelism_threads = 64
        # conf.intra_op_parallelism_threads = 6
        return conf


class ForceModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, FLAGS, *args, **kwargs):
        self.FLAGS = FLAGS
        self.runner = ForceRunner(FLAGS)

    @abstractmethod
    def build(self, *args, **kwargs):
        raise NotImplementedError('Calling an abstract method.')

    def get_scaffold(self):
        max_to_keep = getattr(self.FLAGS, 'max_to_keep', 5)
        load_checkpoint_dir = getattr(self.FLAGS, 'load_checkpoint_dir', None)

        var_list = [v for v in variables._all_saveable_objects() if 'stop_flag' not in v.name]
        saver = tf.train.Saver(var_list=var_list, max_to_keep=max_to_keep, sharded=True)
        ready_for_local_init_op = tf.report_uninitialized_variables(var_list)
        nostore_var_list = list(set(tf.global_variables()) - set(var_list))

        local_init_op = tf.group(tf.variables_initializer(nostore_var_list),
                                 tf.local_variables_initializer(),
                                 tf.tables_initializer(),
                                 name='local_init_op_group')

        if not load_checkpoint_dir:
            return tf.train.Scaffold(local_init_op=local_init_op, ready_for_local_init_op=ready_for_local_init_op, saver=saver)
        else:
            print('load checkpoint from:{}'.format(load_checkpoint_dir))
            ckpt = tf.train.get_checkpoint_state(load_checkpoint_dir)
            if ckpt is None:
                print('no checkpoint at:{}'.format(load_checkpoint_dir))
                return tf.train.Scaffold(local_init_op=local_init_op, ready_for_local_init_op=ready_for_local_init_op, saver=saver)
            else:
                # variables_to_restore = [v for v in tf.contrib.framework.get_variables_to_restore() if 'stop_flag' not in v.name]
                # variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['stop_flag:0'])
                init_assign_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(ckpt.model_checkpoint_path,
                                                                                             var_list,
                                                                                             ignore_missing_vars=True)

                def init_fn(_scaffold, sess):
                    sess.run(init_assign_op, init_feed_dict)
                scaffold = tf.train.Scaffold(init_fn=init_fn,
                                             ready_for_local_init_op=ready_for_local_init_op,
                                             local_init_op=local_init_op,
                                             saver=saver)
                return scaffold

    def get_hooks(self):
        hooks = []
        loss = getattr(self, 'loss', None)
        if loss is not None:
            nan_hook = tf.train.NanTensorHook(loss)
            hooks.append(nan_hook)
        return hooks

    @staticmethod
    def get_chief_only_hooks():
        return None

    @staticmethod
    def get_schedule_config():
        conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        conf.gpu_options.allow_growth = True
        conf.inter_op_parallelism_threads = 64
        # conf.intra_op_parallelism_threads = 6
        return conf

