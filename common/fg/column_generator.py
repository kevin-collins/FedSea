import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers.feature_column import _EmbeddingColumn
from tensorflow.contrib.layers.python.layers import initializers
from collections import OrderedDict
from common.fg import custom_feature_column


def get_init_op(conf_dict):
    name = conf_dict.get('init_op', 'zero')
    if name.lower() == 'zero':
        initializer = tf.zeros_initializer
    elif name.lower == 'one':
        initializer = tf.ones_initializer
    elif name.lower() == 'xavier':
        initializer = initializers.xavier_initializer()
    elif name.lower() == 'var_scale':
        initializer = initializers.variance_scaling_initializer()
    elif name.lower() == 'constant':
        initializer = tf.constant_initializer(conf_dict.get('constant_init', 0.0001))
    elif name.lower() == 'random':
        initializer = tf.random_uniform_initializer
    elif name.lower() == 'normal':
        initializer = tf.truncated_normal_initializer
    else:
        initializer = tf.zeros_initializer
    return initializer


def get_normalizer(conf_dict):
    if 'normalizer' not in conf_dict:
        return None
    params = dict([kv.split('=') for kv in conf_dict['normalizer'].strip().split(',')])
    name = params['method'].lower()
    if name == 'log1p':
        return lambda x: tf.log1p(tf.nn.relu(x))
    elif name == 'l2_norm':
        return lambda x: tf.nn.l2_normalize(x, dim=-1)
    elif name == 'zscore':
        mean, stddev = float(params['mean']), float(params['stddev'])
        if 'clip_min' in params and 'clip_max' in params:
            clip_min, clip_max = float(params['clip_min']), float(params['clip_max'])
            return lambda x: tf.clip_by_value(tf.div((x - mean), stddev), clip_min, clip_max)
        elif 'min' in params and 'max' in params:
            clip_min, clip_max = float(params['min']), float(params['max'])
            return lambda x: tf.div((tf.clip_by_value(x, clip_min, clip_max) - mean), stddev)
        else:
            return lambda x: tf.div((x - mean), stddev)
    elif name == 'zeros':
        return lambda x: tf.zeros_like(x)
    else:
        return None


def real_valued_column(conf_dict, column_dict, input_column=None):
    name = conf_dict.get('column_name', conf_dict['feature_name'])
    column = layers.real_valued_column(
        column_name=name,
        dimension=conf_dict.get('value_dimension', 1),
        default_value=[0.0 for _ in range(int(conf_dict.get('value_dimension', 1)))],
        normalizer=get_normalizer(conf_dict)
    )
    column_dict[name] = column
    return column


def embedding_column(conf_dict, column_dict, input_column=None):
    if 'embedding_dimension' not in conf_dict:
        raise Exception('embedding_dimension not found in config')
    name = conf_dict.get('column_name', conf_dict['feature_name'])
    if 'hash_bucket_size' in conf_dict and 'boundaries' not in conf_dict:
        if isinstance(conf_dict['hash_bucket_size'], list):
            # for cascade embedding column, don't used in pipe
            cascade_column_list = []
            for i, hash_bucket_size in enumerate(conf_dict['hash_bucket_size']):
                # map str/int value to an integer ID which is hash value
                input_column = layers.sparse_column_with_hash_bucket(
                    column_name='{}_{}'.format(name, i),
                    hash_bucket_size=hash_bucket_size,
                )
                if isinstance(conf_dict['embedding_dimension'], list):
                    dimension = conf_dict['embedding_dimension'][i]
                else:
                    dimension = conf_dict['embedding_dimension']
                cascade_column = _EmbeddingColumn(
                    input_column,
                    dimension=dimension,
                    combiner=conf_dict.get('combiner', 'mean'),
                    # shared_embedding_name='{}_C{}'.format(conf_dict.get('shared_name', 'feature_name'), i),
                    max_norm=conf_dict.get('max_norm', None),
                    initializer=get_init_op(conf_dict)
                )
                cascade_column_list.append(cascade_column)
            column = cascade_column_list
        else:
            if input_column is None:
                input_column = layers.sparse_column_with_hash_bucket(
                    column_name=name,
                    hash_bucket_size=conf_dict['hash_bucket_size'],
                )
            # get embedding
            # similar to tf.contrib.layers.embedding_column(input_column, dimension=xx)
            column = _EmbeddingColumn(
                input_column,
                dimension=conf_dict['embedding_dimension'],
                combiner=conf_dict.get('combiner', 'mean'),
                # shared_embedding_name=conf_dict.get('shared_name', None),
                max_norm=conf_dict.get('max_norm', None),
                initializer=get_init_op(conf_dict)
            )
    elif 'boundaries' in conf_dict:
        if input_column is None:
            input_column = layers.real_valued_column(
                column_name=name,
                dimension=conf_dict.get('value_dimension', 1),
                default_value=[0.0 for _ in range(int(conf_dict.get('value_dimension', 1)))]
            )
        column = custom_feature_column.embedding_bucketized_column(
            input_column,
            boundaries=[float(b) for b in conf_dict['boundaries'].split(',')],
            embedding_dimension=conf_dict["embedding_dimension"],
            max_norm=conf_dict.get('max_norm', None),
            initializer=get_init_op(conf_dict),
            is_local=True,
            shared_name=None,
            # shared_name=conf_dict.get('shared_name', None),
            add_random=conf_dict.get('add_random', False)
        )
    else:
        raise Exception('unsupported embedding column configuration')
    column_dict[name] = column
    return column


def embedding_variable_column(conf_dict, column_dict, input_column=None):
    if 'embedding_dimension' not in conf_dict:
        raise Exception('embedding_dimension not found in config')
    name = conf_dict.get('column_name', conf_dict['feature_name'])
    if 'boundaries' in conf_dict:
        if input_column is None:
            input_column = layers.real_valued_column(
                column_name=name,
                dimension=conf_dict.get('value_dimension', 1),
                default_value=[0.0 for _ in range(int(conf_dict.get('value_dimension', 1)))]
            )
        column = custom_feature_column.embedding_bucketized_column(
            input_column,
            boundaries=[float(b) for b in conf_dict['boundaries'].split(',')],
            embedding_dimension=conf_dict["embedding_dimension"],
            max_norm=conf_dict.get('max_norm', None),
            initializer=get_init_op(conf_dict),
            is_local=True,
            shared_name=None,
            # shared_name=conf_dict.get('shared_name', None),
            add_random=conf_dict.get('add_random', False)
        )
    else:
        if input_column is None:
            input_column = tf.contrib.layers.sparse_column_with_embedding(column_name=name,
                                                                          dtype=tf.string,
                                                                          partition_num=5,
                                                                          steps_to_live=2000000)
        column = _EmbeddingColumn(
            input_column,
            dimension=conf_dict['embedding_dimension'],
            combiner=conf_dict.get('combiner', 'mean'),
            # shared_embedding_name=conf_dict.get('shared_name', None),
            max_norm=conf_dict.get('max_norm', None),
            initializer=get_init_op(conf_dict)
        )
        column_dict[name] = column
    column_dict[name] = column
    return column


def sequence_column(conf_dict, column_dict):
    if 'sequence_name' not in conf_dict:
        raise Exception('sequence_name not found in config')
    seq_len_dict = OrderedDict()
    seq_column_dict = OrderedDict()
    sequence_name = conf_dict['sequence_name']
    for att_fea in conf_dict['features']:
        att_seq_name = '{}_{}'.format(sequence_name, att_fea['feature_name'])
        seq_len_dict[att_seq_name] = conf_dict['sequence_length']
        att_fea['column_name'] = att_seq_name
        if 'init_op' not in att_fea:
            att_fea['init_op'] = conf_dict.get('init_op', None)
        globals()[att_fea['transform_name']](att_fea, seq_column_dict)
        column_dict[att_seq_name] = seq_column_dict[att_seq_name]
    column_dict[sequence_name] = [seq_column_dict, seq_len_dict]


def kv_column(conf_dict, column_dict):
    if 'embedding_dimension' not in conf_dict:
        raise Exception('embedding_dimension not found in config, must be set for kv_column')

    kv_column_dict = OrderedDict()
    name = conf_dict.get('column_name', conf_dict['feature_name'])
    conf_dict['column_name'] = '{}_key'.format(name)
    embedding_column(conf_dict, kv_column_dict)
    key_column = kv_column_dict[conf_dict['column_name']]

    conf_dict['column_name'] = '{}_value'.format(name)
    real_valued_column(conf_dict, kv_column_dict)
    value_column = kv_column_dict[conf_dict['column_name']]

    column_dict[name] = [('{}_key'.format(name), key_column),
                         ('{}_value'.format(name), value_column),
                         conf_dict]


def kvar_column(conf_dict, column_dict):
    if 'embedding_dimension' not in conf_dict:
        raise Exception('embedding_dimension not found in config, must be set for kv_column')

    kv_column_dict = OrderedDict()
    name = conf_dict.get('column_name', conf_dict['feature_name'])
    conf_dict['column_name'] = '{}_key'.format(name)
    embedding_variable_column(conf_dict, kv_column_dict)
    key_column = kv_column_dict[conf_dict['column_name']]

    conf_dict['column_name'] = '{}_value'.format(name)
    real_valued_column(conf_dict, kv_column_dict)
    value_column = kv_column_dict[conf_dict['column_name']]

    column_dict[name] = [('{}_key'.format(name), key_column),
                         ('{}_value'.format(name), value_column),
                         conf_dict]


def bucketized_column(conf_dict, column_dict, input_column=None):
    name = conf_dict.get('column_name', conf_dict['feature_name'])
    column = layers.bucketized_column(
        input_column,
        boundaries=[float(b) for b in conf_dict['boundaries'].split(',')]
    )
    column_dict[name] = column
    return column


def one_hot_column(conf_dict, column_dict, input_column=None):
    name = conf_dict.get('column_name', conf_dict['feature_name'])
    column = layers.sparse_column_with_hash_bucket(
        column_name=name,
        hash_bucket_size=conf_dict['hash_bucket_size'],
    )
    column_dict[name] = column
    return column


def pipe(conf_dict, column_dict):
    input_column = None
    for conf in conf_dict['pipe']:
        pipe_conf_dict = conf_dict.copy()
        pipe_conf_dict.update(conf)
        input_column = globals()[pipe_conf_dict['transform_name']](pipe_conf_dict, column_dict, input_column)


def cross(conf_dict, column_dict, input_column=None):
    name = conf_dict.get('column_name', conf_dict['feature_name'])
    if input_column is None:
        input_column = layers.crossed_column(
            [column_dict[conf_dict['cross'][0]], column_dict[conf_dict['cross'][1]]],
            hash_bucket_size=conf_dict['hash_bucket_size']
        )
    column = _EmbeddingColumn(
        input_column,
        dimension=conf_dict.get["embedding_dimension"],
        combiner=conf_dict.get('combiner', 'mean'),
        # shared_embedding_name=conf_dict.get('shared_name', None),
        max_norm=conf_dict.get('max_norm', None),
        initializer=get_init_op(conf_dict)
    )
    column_dict[name] = column
    return column


def KvColumn(conf_dict, column_dict):
    input_name = conf_dict.get('column_name', conf_dict['feature_name'])

    value_column = tf.contrib.layers.real_valued_column(column_name = input_name + '_value',
                                                        dtype=tf.float32,
                                                        default_value=0.0,
                                                        dimension=1)

    column_dict[input_name+'_value'] = value_column

    conf_dict['feature_name'] = input_name + '_id'
    embedding_column(conf_dict, column_dict)
    conf_dict['feature_name'] = input_name
