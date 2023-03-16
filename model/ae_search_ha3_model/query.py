from tensorflow.contrib import layers
from common.model_ops.attention import attention
from utils.util import get_part_embedding, get_gaussian_embedding
import tensorflow as tf


class Query(object):

    def __init__(self, column_builder):
        # query term column
        # self.query_column = column_builder.get_column_list(['keyword_hash'])
        # self.origin_query_column = column_builder.get_column_list(['origin_keyword_hash'])
        # self.query_term_column = column_builder.get_column_list(['term%d_hash' % i for i in range(1, 6, 1)])
        # self.origin_term_column = column_builder.get_column_list(['origin_term%d_hash' % i for i in range(1, 6, 1)])
        # self.query_length_column = [layers.real_valued_column(column_name='query_length', dimension=1, default_value=0.0)]
        # self.origin_query_length_column = [layers.real_valued_column(column_name='origin_query_length', dimension=1, default_value=0.0)]
        self.query_column = column_builder.get_column_list(['feature_4'])
        self.origin_query_column = column_builder.get_column_list(['feature_6'])
        self.query_term_column = column_builder.get_column_list(['feature_5_%d' % i for i in range(1, 6, 1)])
        self.origin_term_column = column_builder.get_column_list(['feature_7_%d' % i for i in range(1, 6, 1)])
        self.query_length_column = [
            layers.real_valued_column(column_name='query_length', dimension=1, default_value=0.0)]
        self.origin_query_length_column = [
            layers.real_valued_column(column_name='origin_query_length', dimension=1, default_value=0.0)]

    @staticmethod
    def update_query_length(features):
        query_length, origin_query_length = 0, 0
        term_list, origin_term_list = [], []
        for i in range(1, 6, 1):
            # term_list.append(features['term{}_hash'.format(i)])
            # origin_term_list.append(features['origin_term{}_hash'.format(i)])
            term_list.append(features['feature_5_{}'.format(i)])
            origin_term_list.append(features['feature_7_{}'.format(i)])

        terms = tf.sparse_concat(axis=1, sp_inputs=term_list)
        terms = tf.sparse_to_dense(terms.indices, terms.dense_shape, terms.values, default_value='-911')
        terms = tf.to_float(tf.equal(terms, '-911'))
        query_length += tf.reduce_sum(terms, axis=-1)

        origins = tf.sparse_concat(axis=1, sp_inputs=origin_term_list)
        origins = tf.sparse_to_dense(origins.indices, origins.dense_shape, origins.values, default_value='-911')
        origins = tf.to_float(tf.equal(origins, '-911'))
        origin_query_length += tf.reduce_sum(origins, axis=-1)

        query_length = 5 - query_length
        origin_query_length = 5 - origin_query_length
        features.update({'query_length': query_length, 'origin_query_length': origin_query_length})

    def get_query_token_layer(self, features, scope=None):
        query_input_layer = layers.input_from_feature_columns(features, self.query_column, scope=scope)
        origin_query_input_layer = layers.input_from_feature_columns(features, self.origin_query_column, scope=scope)
        query_term_input_layer = layers.input_from_feature_columns(features, self.query_term_column, scope=scope)
        origin_term_input_layer = layers.input_from_feature_columns(features, self.origin_term_column, scope=scope)
        query_length = layers.input_from_feature_columns(features, self.query_length_column, scope=scope)
        origin_query_length = layers.input_from_feature_columns(features, self.origin_query_length_column, scope=scope)

        # mask query & origin query
        query_input_layer = tf.where(tf.squeeze(tf.greater(query_length, 0), axis=1),
                                     query_input_layer, tf.zeros_like(query_input_layer))
        origin_query_input_layer = tf.where(tf.squeeze(tf.greater(origin_query_length, 0), axis=1),
                                            origin_query_input_layer, tf.zeros_like(origin_query_input_layer))

        query_input_layer = tf.expand_dims(query_input_layer, axis=1)
        origin_query_input_layer = tf.expand_dims(origin_query_input_layer, axis=1)

        # mask query term
        shape = query_term_input_layer.get_shape().as_list()[-1]
        query_term_input_layer = tf.reshape(query_term_input_layer, [-1, 5, shape // 5])
        masks = tf.sequence_mask(query_length, 5)

        items_2d = tf.reshape(query_term_input_layer, [-1, tf.shape(query_term_input_layer)[2]])
        query_term_input_layer = tf.reshape(tf.where(tf.reshape(masks, [-1]), items_2d, tf.zeros_like(items_2d)),
                                            tf.shape(query_term_input_layer))
        # mask origin query term
        shape = origin_term_input_layer.get_shape().as_list()[-1]
        origin_term_input_layer = tf.reshape(origin_term_input_layer, [-1, 5, shape // 5])
        masks = tf.sequence_mask(origin_query_length, 5)

        items_2d = tf.reshape(origin_term_input_layer, [-1, tf.shape(origin_term_input_layer)[2]])
        origin_term_input_layer = tf.reshape(tf.where(tf.reshape(masks, [-1]), items_2d, tf.zeros_like(items_2d)),
                                             tf.shape(origin_term_input_layer))

        term_vec, _ = attention(queries=query_input_layer,
                                queries_length=None,
                                keys=query_term_input_layer,
                                keys_length=query_length,
                                query_masks=None,
                                key_masks=None)
        origin_term_vec, _ = attention(queries=origin_query_input_layer,
                                       queries_length=None,
                                       keys=origin_term_input_layer,
                                       keys_length=origin_query_length,
                                       query_masks=None,
                                       key_masks=None)
        query_token_layer = tf.concat([query_input_layer, origin_query_input_layer, term_vec, origin_term_vec], axis=1)
        return query_token_layer

    def get_part_query_token_layer(self, features, part, scope=None, stop_gradient=False, isolate=False):
        query_input_layer = layers.input_from_feature_columns(features, self.query_column, scope=scope)
        origin_query_input_layer = layers.input_from_feature_columns(features, self.origin_query_column, scope=scope)
        query_term_input_layer = layers.input_from_feature_columns(features, self.query_term_column, scope=scope)
        origin_term_input_layer = layers.input_from_feature_columns(features, self.origin_term_column, scope=scope)
        query_length = layers.input_from_feature_columns(features, self.query_length_column, scope=scope)
        origin_query_length = layers.input_from_feature_columns(features, self.origin_query_length_column, scope=scope)

        query_input_layer = get_part_embedding(query_input_layer, part, stop_gradient=stop_gradient, isolate=isolate)
        origin_query_input_layer = get_part_embedding(origin_query_input_layer, part, stop_gradient=stop_gradient, isolate=isolate)
        query_term_input_layer = get_part_embedding(query_term_input_layer, part, stop_gradient=stop_gradient, isolate=isolate)
        origin_term_input_layer = get_part_embedding(origin_term_input_layer, part, stop_gradient=stop_gradient, isolate=isolate)

        # mask query & origin query
        query_input_layer = tf.where(tf.squeeze(tf.greater(query_length, 0), axis=1),
                                     query_input_layer, tf.zeros_like(query_input_layer))
        origin_query_input_layer = tf.where(tf.squeeze(tf.greater(origin_query_length, 0), axis=1),
                                            origin_query_input_layer, tf.zeros_like(origin_query_input_layer))

        query_input_layer = tf.expand_dims(query_input_layer, axis=1)
        origin_query_input_layer = tf.expand_dims(origin_query_input_layer, axis=1)

        # mask query term
        shape = query_term_input_layer.get_shape().as_list()[-1]
        query_term_input_layer = tf.reshape(query_term_input_layer, [-1, 5, shape // 5])
        masks = tf.sequence_mask(query_length, 5)

        items_2d = tf.reshape(query_term_input_layer, [-1, tf.shape(query_term_input_layer)[2]])
        query_term_input_layer = tf.reshape(tf.where(tf.reshape(masks, [-1]), items_2d, tf.zeros_like(items_2d)),
                                            tf.shape(query_term_input_layer))
        # mask origin query term
        shape = origin_term_input_layer.get_shape().as_list()[-1]
        origin_term_input_layer = tf.reshape(origin_term_input_layer, [-1, 5, shape // 5])
        masks = tf.sequence_mask(origin_query_length, 5)

        items_2d = tf.reshape(origin_term_input_layer, [-1, tf.shape(origin_term_input_layer)[2]])
        origin_term_input_layer = tf.reshape(tf.where(tf.reshape(masks, [-1]), items_2d, tf.zeros_like(items_2d)),
                                             tf.shape(origin_term_input_layer))

        term_vec, _ = attention(queries=query_input_layer,
                                queries_length=None,
                                keys=query_term_input_layer,
                                keys_length=query_length,
                                query_masks=None,
                                key_masks=None)
        origin_term_vec, _ = attention(queries=origin_query_input_layer,
                                       queries_length=None,
                                       keys=origin_term_input_layer,
                                       keys_length=origin_query_length,
                                       query_masks=None,
                                       key_masks=None)
        query_token_layer = tf.concat([query_input_layer, origin_query_input_layer, term_vec, origin_term_vec], axis=1)
        return query_token_layer

    def get_specific_query_token_layer(self,
                                       features,
                                       method='default',
                                       is_training=False,
                                       part=0,
                                       scope=None,
                                       stop_gradient=False,
                                       isolate=False,
                                       valid_dimension=8,
                                       embedding_dimension=16,
                                       aux_reg=False
                                       ):
        # default operation for all situations
        query_input_layer = layers.input_from_feature_columns(features, self.query_column, scope=scope)
        origin_query_input_layer = layers.input_from_feature_columns(features, self.origin_query_column, scope=scope)
        query_term_input_layer = layers.input_from_feature_columns(features, self.query_term_column, scope=scope)
        origin_term_input_layer = layers.input_from_feature_columns(features, self.origin_term_column, scope=scope)
        query_length = layers.input_from_feature_columns(features, self.query_length_column, scope=scope)
        origin_query_length = layers.input_from_feature_columns(features, self.origin_query_length_column, scope=scope)

        if method == 'part':
            query_input_layer = get_part_embedding(query_input_layer,
                                                   part,
                                                   valid_dimension=valid_dimension,
                                                   embedding_dimension=embedding_dimension,
                                                   stop_gradient=stop_gradient,
                                                   isolate=isolate,
                                                   aux_reg=aux_reg)
            origin_query_input_layer = get_part_embedding(origin_query_input_layer,
                                                          part,
                                                          valid_dimension=valid_dimension,
                                                          embedding_dimension=embedding_dimension,
                                                          stop_gradient=stop_gradient,
                                                          isolate=isolate,
                                                          aux_reg=aux_reg)
            query_term_input_layer = get_part_embedding(query_term_input_layer,
                                                        part,
                                                        valid_dimension=valid_dimension,
                                                        embedding_dimension=embedding_dimension,
                                                        stop_gradient=stop_gradient,
                                                        isolate=isolate,
                                                        aux_reg=aux_reg)
            origin_term_input_layer = get_part_embedding(origin_term_input_layer,
                                                         part,
                                                         valid_dimension=valid_dimension,
                                                         embedding_dimension=embedding_dimension,
                                                         stop_gradient=stop_gradient,
                                                         isolate=isolate,
                                                         aux_reg=aux_reg)
        elif method == 'gaussian':
            query_input_layer = get_gaussian_embedding(query_input_layer, is_training=is_training,
                                                       valid_dimension=valid_dimension, embedding_dimension=embedding_dimension)
            origin_query_input_layer = get_gaussian_embedding(origin_query_input_layer, is_training=is_training,
                                                              valid_dimension=valid_dimension, embedding_dimension=embedding_dimension)
            query_term_input_layer = get_gaussian_embedding(query_term_input_layer, is_training=is_training,
                                                            valid_dimension=valid_dimension, embedding_dimension=embedding_dimension)
            origin_term_input_layer = get_gaussian_embedding(origin_term_input_layer, is_training=is_training,
                                                             valid_dimension=valid_dimension, embedding_dimension=embedding_dimension)
        elif method != 'default':
            raise Exception('unsupported embedding method:{}'.format(method))

        # mask query & origin query
        query_input_layer = tf.where(tf.squeeze(tf.greater(query_length, 0), axis=1),
                                     query_input_layer, tf.zeros_like(query_input_layer))
        origin_query_input_layer = tf.where(tf.squeeze(tf.greater(origin_query_length, 0), axis=1),
                                            origin_query_input_layer, tf.zeros_like(origin_query_input_layer))

        query_input_layer = tf.expand_dims(query_input_layer, axis=1)
        origin_query_input_layer = tf.expand_dims(origin_query_input_layer, axis=1)

        # mask query term
        shape = query_term_input_layer.get_shape().as_list()[-1]
        query_term_input_layer = tf.reshape(query_term_input_layer, [-1, 5, shape // 5])
        masks = tf.sequence_mask(query_length, 5)

        items_2d = tf.reshape(query_term_input_layer, [-1, tf.shape(query_term_input_layer)[2]])
        query_term_input_layer = tf.reshape(tf.where(tf.reshape(masks, [-1]), items_2d, tf.zeros_like(items_2d)),
                                            tf.shape(query_term_input_layer))
        # mask origin query term
        shape = origin_term_input_layer.get_shape().as_list()[-1]
        origin_term_input_layer = tf.reshape(origin_term_input_layer, [-1, 5, shape // 5])
        masks = tf.sequence_mask(origin_query_length, 5)

        items_2d = tf.reshape(origin_term_input_layer, [-1, tf.shape(origin_term_input_layer)[2]])
        origin_term_input_layer = tf.reshape(tf.where(tf.reshape(masks, [-1]), items_2d, tf.zeros_like(items_2d)),
                                             tf.shape(origin_term_input_layer))

        term_vec, _ = attention(queries=query_input_layer,
                                queries_length=None,
                                keys=query_term_input_layer,
                                keys_length=query_length,
                                query_masks=None,
                                key_masks=None)
        origin_term_vec, _ = attention(queries=origin_query_input_layer,
                                       queries_length=None,
                                       keys=origin_term_input_layer,
                                       keys_length=origin_query_length,
                                       query_masks=None,
                                       key_masks=None)
        query_token_layer = tf.concat([query_input_layer, origin_query_input_layer, term_vec, origin_term_vec], axis=1)
        return query_token_layer
