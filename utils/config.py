from tensorflow.python.lib.io import file_io
import os
import sys
import json
import tensorflow as tf

current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(current_dir)


def string2kv(s, d1='&', d2='='):
    kv = {}
    if not s:
        return kv
    for e in s.split(d1):
        pair = e.split(d2)
        if len(pair) != 2:
            continue
        kv[pair[0]] = pair[1]
    return kv


def get_absolute_path(file_path):
    if not file_path.startswith(os.sep):
        file_path = os.path.join(root_dir, file_path)
    return file_path


def parse_model_conf(FLAGS):
    # fg / fc / mc config
    if getattr(FLAGS, 'fg_conf', None):
        if FLAGS.task_index == 0:
            file_io.copy(get_absolute_path(FLAGS.fg_conf), os.path.join(FLAGS.checkpointDir, 'fg.json'), overwrite=True)
        with open(get_absolute_path(FLAGS.fg_conf)) as fg:
            FLAGS.fg_conf = json.load(fg)
    else:
        tf.logging.warn('fg_conf not defined in FLAGS')

    if getattr(FLAGS, 'fc_conf', None):
        with open(get_absolute_path(FLAGS.fc_conf)) as fc:
            FLAGS.fc_conf = json.load(fc)
    else:
        tf.logging.warn('fc_conf not defined in FLAGS')

    if getattr(FLAGS, 'mc_conf', None):
        with open(get_absolute_path(FLAGS.mc_conf)) as mc:
            FLAGS.mc_conf = json.load(mc)
    else:
        tf.logging.warn('mc_conf not defined in FLAGS')



def parse_model_conf_v2(FLAGS):
    # fg / fc / mc config
    conf_path = os.sep.join(FLAGS.global_conf.split(os.sep)[:-1])
    if getattr(FLAGS, 'fg_conf', None):
        fg_path = get_absolute_path(os.path.join(conf_path, FLAGS.fg_conf))
        if FLAGS.task_index == 0:
            file_io.copy(fg_path, os.path.join(FLAGS.checkpointDir, 'fg.json'), overwrite=True)
        with open(fg_path) as fg:
            FLAGS.fg_conf = json.load(fg)
    else:
        tf.logging.warn('fg_conf not defined in FLAGS')

    if getattr(FLAGS, 'fc_conf', None):
        with open(get_absolute_path(os.path.join(conf_path, FLAGS.fc_conf))) as fc:
            FLAGS.fc_conf = json.load(fc)
    else:
        tf.logging.warn('fc_conf not defined in FLAGS')

    if getattr(FLAGS, 'mc_conf', None):
        with open(get_absolute_path(os.path.join(conf_path, FLAGS.mc_conf))) as mc:
            FLAGS.mc_conf = json.load(mc)
    else:
        tf.logging.warn('mc_conf not defined in FLAGS')

def get_table_size(odps_table):
    reader = tf.python_io.TableReader(odps_table)
    total_records_num = reader.get_row_count()
    return total_records_num


def is_string(value):
    if sys.version_info.major < 3:
        return isinstance(value, (str, unicode))
    else:
        return isinstance(value, (str, bytes))


def add_define_func(name, value=None, comment=None):
    if is_string(value):
        getattr(tf.app.flags, 'DEFINE_string')(name, value, comment)
    elif isinstance(value, bool):
        getattr(tf.app.flags, 'DEFINE_boolean')(name, value, comment)
    elif isinstance(value, int):
        getattr(tf.app.flags, 'DEFINE_integer')(name, value, comment)
    elif isinstance(value, float):
        getattr(tf.app.flags, 'DEFINE_float')(name, value, comment)
    elif isinstance(value, list):
        fn = getattr(tf.app.flags, 'DEFINE_list', None)
        if fn is not None:
            fn(name, value, comment)
    else:
        pass


def parse_global_conf(FLAGS):
    conf_path = FLAGS.global_conf
    if not conf_path.startswith(os.sep):
        conf_path = os.path.join(root_dir, conf_path)
    if not os.path.exists(conf_path):
        raise Exception('global_conf:{} not exists.'.format(conf_path))
    with open(conf_path, 'r') as cf:
        global_conf = json.load(cf)

    for params in global_conf.values():
        for name, value in params.items():
            if getattr(FLAGS, name, None) is None:
                add_define_func(name, value)
            setattr(FLAGS, name, value)

