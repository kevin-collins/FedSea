import re
import traceback
from tensorflow.contrib.layers.python.layers import initializers
import tensorflow as tf
from tensorflow import logging
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers.feature_column_ops import _Transformer
from tensorflow.contrib.layers.python.layers.feature_column_ops import _add_variable_collection
from tensorflow.contrib.layers.python.layers.feature_column_ops import _create_embedding_lookup
from tensorflow.contrib.layers.python.layers.feature_column_ops import _log_variable
from tensorflow.contrib.layers.python.layers.feature_column_ops import _maybe_reshape_input_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope


def partitioner(ps_num, mem=8 * 1024 * 1024):
    linear_partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=ps_num,
        min_slice_size=mem)
    return linear_partitioner


def combine_parted_variables(vs, combine_postfix='/combine'):
    mmap = {}
    for x in vs:
        mkey = re.sub('/part_[0-9]*:', '/', x.name)
        if mkey in mmap:
            mmap[mkey].append(x)
        else:
            mmap[mkey] = [x]
    return [tf.concat(t, 0, name=k.replace(':', '_') + combine_postfix) for k, t in mmap.items()]


def add_embed_layer_norm(layer_tensor, columns):
    if layer_tensor is None:
        return
    i = 0
    for column in sorted(set(columns), key=lambda x: x.key):
        try:
            dim = column.dimension
        except:
            dim = column.embedding_dimension
        tf.summary.scalar(name=column.name, tensor=tf.reduce_mean(tf.norm(layer_tensor[:, i:i + dim], axis=-1)))
        i += dim


def add_norm2_summary(collection_name, summary_prefix="Norm2/", contain_string=""):
    variables = tf.get_collection(collection_name)
    vv = combine_parted_variables(variables)
    for x in vv:
        if contain_string in x.name:
            logging.info("add norm2 %s to summary with shape %s" % (str(x.name), str(x.shape)))
            try:
                tf.summary.scalar(name=summary_prefix + x.name.replace(":", "_"), tensor=tf.norm(x))
            except:
                tf.summary.scalar(name=summary_prefix + x.name.replace(":", "_"), tensor=tf.norm(x, axis=[-2, -1]))


def add_weight_summary(collection_name, summary_prefix="Weight/", contain_string=""):
    variables = tf.get_collection(collection_name)
    vv = combine_parted_variables(variables)
    for x in vv:
        if contain_string in x.name:
            try:
                name = x.name.replace(":", "_")
                x = tf.reshape(x, [-1])
                logging.info("add weight %s to summary with shape %s" % (str(x.name), str(x.shape)))

                tf.summary.scalar(name=summary_prefix + "Norm2/" + name,
                                  tensor=tf.norm(x, axis=-1))
                tf.summary.histogram(name=summary_prefix + "Hist/" + name,
                                     values=x)
                mean, variance = tf.nn.moments(x, axes=0)
                tf.summary.scalar(name=summary_prefix + "Mean/" + name,
                                  tensor=mean)
                tf.summary.scalar(name=summary_prefix + "Variance/" + name,
                                  tensor=variance)
                tf.summary.scalar(name=summary_prefix + "PosRatio/" + name, tensor=greater_zero_fraction(x))
            except Exception as e:
                logging.warn('Got exception run : %s | %s' % (e, traceback.format_exc()))
                logging.warn("add_dense_output_summary with rank not 2: [%s],shape=[%s]" % (str(x.name), str(x.shape)))


def add_dense_output_summary(collection_name, summary_prefix="DenseOutput/", contain_string=""):
    variables = tf.get_collection(collection_name)
    vv = combine_parted_variables(variables)
    for x in vv:
        if contain_string in x.name:
            try:
                logging.info("add dense_output %s to summary with shape %s" % (str(x.name), str(x.shape)))
                if len(x.shape) == 3:
                    tf.summary.scalar(name=summary_prefix + "Norm2/" + x.name.replace(":", "_"),
                                      tensor=tf.reduce_mean(tf.norm(x)))
                    tf.summary.histogram(name=summary_prefix + "AbsHistMax/" + x.name.replace(":", "_"),
                                         values=tf.reduce_max(tf.abs(x), axis=-1))
                    mean, variance = tf.nn.moments(x, axes=[0, 1])
                    tf.summary.scalar(name=summary_prefix + "Mean/" + x.name.replace(":", "_"),
                                      tensor=tf.reduce_mean(mean))
                    tf.summary.scalar(name=summary_prefix + "Variance/" + x.name.replace(":", "_"),
                                      tensor=tf.reduce_mean(variance))
                    tf.summary.histogram(name=summary_prefix + "PosR/" + x.name.replace(":", "_"),
                                         values=greater_zero_histogram(x))
                elif x.shape[1] > 1:
                    tf.summary.scalar(name=summary_prefix + "Norm2/" + x.name.replace(":", "_"),
                                      tensor=tf.reduce_mean(tf.norm(x, axis=-1)))
                    tf.summary.histogram(name=summary_prefix + "AbsHistMax/" + x.name.replace(":", "_"),
                                         values=tf.reduce_max(tf.abs(x), axis=-1))
                    mean, variance = tf.nn.moments(x, axes=-1)
                    tf.summary.scalar(name=summary_prefix + "Mean/" + x.name.replace(":", "_"),
                                      tensor=tf.reduce_mean(mean))
                    tf.summary.scalar(name=summary_prefix + "Variance/" + x.name.replace(":", "_"),
                                      tensor=tf.reduce_mean(variance))
                    tf.summary.histogram(name=summary_prefix + "PosR/" + x.name.replace(":", "_"),
                                         values=greater_zero_histogram(x))
                else:
                    tf.summary.scalar(name=summary_prefix + "Norm2/" + x.name.replace(":", "_"),
                                      tensor=tf.reduce_mean(tf.norm(x, axis=-1)))
                    tf.summary.histogram(name=summary_prefix + "AbsHistMax/" + x.name.replace(":", "_"),
                                         values=tf.reduce_max(tf.abs(x)))
                    mean, variance = tf.nn.moments(x, axes=0)
                    tf.summary.scalar(name=summary_prefix + "Mean/" + x.name.replace(":", "_"),
                                      tensor=mean[0])
                    tf.summary.scalar(name=summary_prefix + "Variance/" + x.name.replace(":", "_"),
                                      tensor=variance[0])
                tf.summary.scalar(name=summary_prefix + "PosRatio/" + x.name.replace(":", "_"),
                                  tensor=greater_zero_fraction(x))
            except Exception as e:
                logging.warn('Got exception run : %s | %s' % (e, traceback.format_exc()))
                logging.warn("add_dense_output_summary with rank not 2: [%s],shape=[%s]" % (str(x.name), str(x.shape)))


# overall fraction
def greater_zero_fraction(value, name=None):
    with tf.name_scope(name, "greater_fraction", [value]):
        value = tf.convert_to_tensor(value, name="value")
        zero = tf.constant(0, dtype=value.dtype, name="zero")
        return math_ops.reduce_mean(
            math_ops.cast(math_ops.greater(value, zero), tf.float32))


# histogram of each sample's zero fraction
def greater_zero_histogram(value, name=None):
    with tf.name_scope(name, "greater_histogram", [value]):
        value = tf.convert_to_tensor(value, name="value")
        zero = tf.constant(0, dtype=value.dtype, name="zero")
        return math_ops.reduce_mean(
            math_ops.cast(math_ops.greater(value, zero), tf.float32), axis=-1)


def add_layer_summary(value, tag):
    tf.summary.scalar("%s/fraction_of_zero_values" % tag, tf.nn.zero_fraction(value))
    tf.summary.histogram("%s/activation" % tag, value)


def add_logits_summary(name, tensor):
    tf.summary.scalar("%s/scalar/mean" % name, tf.reduce_mean(tensor))
    tf.summary.scalar("%s/scalar/max" % name, tf.reduce_max(tensor))
    tf.summary.scalar("%s/scalar/min" % name, tf.reduce_min(tensor))
    tf.summary.histogram("%s/histogram" % name, tensor)


def model_arg_scope(weight_decay=0.0005, weights_initializer=initializers.xavier_initializer(),
                    biases_initializer=init_ops.zeros_initializer()):
    with arg_scope(
            [layers.fully_connected, layers.conv2d],
            weights_initializer=weights_initializer,
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            biases_initializer=biases_initializer) as arg_sc:
        return arg_sc


def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor.

  Args:
      params: A 2D tensor.
      indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
      name: A name for the operation (optional).

  Returns:
      A 2D Tensor. Has the same type as ``params``.
  """
    with tf.name_scope(name, "gather_cols", values=[params, indices]) as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        i_shape = indices._shape_as_list()
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(tf.gather(p_flat, i_flat),
                          [p_shape[0], i_shape[0]])


# for print debug
def weighted_sum_from_feature_columns(columns_to_tensors,
                                      feature_columns,
                                      num_outputs,
                                      weight_collections=None,
                                      trainable=True,
                                      scope=None):
    """A tf.contrib.layers style linear prediction builder based on FeatureColumn.

  Generally a single example in training data is described with feature columns.
  This function generates weighted sum for each num_outputs. Weighted sum refers
  to logits in classification problems. It refers to prediction itself for
  linear regression problems.

  Example:

    ```
    # Building model for training
    feature_columns = (
        real_valued_column("my_feature1"),
        ...
    )
    columns_to_tensor = tf.parse_example(...)
    logits = weighted_sum_from_feature_columns(
        columns_to_tensors=columns_to_tensor,
        feature_columns=feature_columns,
        num_outputs=1)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                   logits=logits)
    ```

  Args:
    columns_to_tensors: A mapping from feature column to tensors. 'string' key
      means a base feature (not-transformed). It can have FeatureColumn as a
      key too. That means that FeatureColumn is already transformed by input
      pipeline. For example, `inflow` may have handled transformations.
    feature_columns: A set containing all the feature columns. All items in the
      set should be instances of classes derived from FeatureColumn.
    num_outputs: An integer specifying number of outputs. Default value is 1.
    weight_collections: List of graph collections to which weights are added.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for variable_scope.

  Returns:
    A tuple containing:

      * A Tensor which represents predictions of a linear model.
      * A dictionary which maps feature_column to corresponding Variable.
      * A Variable which is used for bias.

  Raises:
    ValueError: if FeatureColumn cannot be used for linear predictions.
  """
    columns_to_tensors = columns_to_tensors.copy()
    with variable_scope.variable_scope(
            scope,
            default_name='weighted_sum_from_feature_columns',
            values=columns_to_tensors.values()):
        output_tensors = []
        column_to_variable = dict()
        print_variable = dict()
        transformer = _Transformer(columns_to_tensors)
        # pylint: disable=protected-access
        for column in sorted(set(feature_columns), key=lambda x: x.key):
            transformed_tensor = transformer.transform(column)
            try:
                embedding_lookup_arguments = column._wide_embedding_lookup_arguments(
                    transformed_tensor)
                variable, predictions = _create_embedding_lookup(
                    column,
                    columns_to_tensors,
                    embedding_lookup_arguments,
                    num_outputs,
                    trainable,
                    weight_collections)
            except NotImplementedError:
                with variable_scope.variable_scope(
                        None,
                        default_name=column.name,
                        values=columns_to_tensors.values()):
                    tensor = column._to_dense_tensor(transformed_tensor)
                    tensor = _maybe_reshape_input_tensor(
                        tensor, column.name, output_rank=2)
                    variable = [
                        contrib_variables.model_variable(
                            name='weight',
                            shape=[tensor.get_shape()[1], num_outputs],
                            initializer=init_ops.zeros_initializer(),
                            trainable=trainable,
                            collections=weight_collections)
                    ]
                    predictions = math_ops.matmul(tensor, variable[0], name='matmul')
            except ValueError as ee:
                raise ValueError('Error creating weighted sum for column: {}.\n'
                                 '{}'.format(column.name, ee))
            output_tensors.append(array_ops.reshape(
                predictions, shape=(-1, num_outputs)))
            column_to_variable[column] = variable
            print_variable[column.name] = variable[0]
            _log_variable(variable)
            checkpoint_path = column._checkpoint_path()
            if checkpoint_path is not None:
                path, tensor_name = checkpoint_path
                weights_to_restore = variable
                if len(variable) == 1:
                    weights_to_restore = variable[0]
                checkpoint_utils.init_from_checkpoint(path,
                                                      {tensor_name: weights_to_restore})
                # fc._maybe_restore_from_checkpoint(column._checkpoint_path(), variable)  # pylint: disable=protected-access
        # pylint: enable=protected-access
        predictions_no_bias = math_ops.add_n(output_tensors)
        bias = contrib_variables.model_variable(
            'bias_weight',
            shape=[num_outputs],
            initializer=init_ops.zeros_initializer(),
            trainable=trainable,
            collections=_add_variable_collection(weight_collections))
        _log_variable(bias)
        predictions = nn_ops.bias_add(predictions_no_bias, bias)

        return predictions, column_to_variable, bias, output_tensors, print_variable


from tensorflow.python.framework import ops


def my_weighted_cross_entropy_with_logits(neg_num, pos_num, logits, name=None):
    """Computes a weighted cross entropy.


  """
    with ops.name_scope(name, "logistic_loss", [logits, neg_num, pos_num]) as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        neg_num = ops.convert_to_tensor(neg_num, name="neg_num")
        pos_num = ops.convert_to_tensor(pos_num, name="pos_num")
        try:
            neg_num.get_shape().merge_with(logits.get_shape())
            pos_num.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError(
                "logits and targets must have the same shape (%s vs %s vs %s)" %
                (logits.get_shape(), neg_num.get_shape(), pos_num.get_shape()))

        # The logistic loss formula from above is
        #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
        # For x < 0, a more numerically stable formula is
        #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(x)) - l * x
        # To avoid branching, we use the combined version
        #   (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))

        return math_ops.add(
            neg_num * logits,
            (neg_num + pos_num) * (math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) +
                                   nn_ops.relu(-logits)),
            name=name)
