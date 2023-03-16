from common.fg import column_generator


class FeatureColumnBuilder(object):
    def __init__(self, FLAGS, fg_conf, fc_conf, **kwargs):
        self._FLAGS = FLAGS
        self._feature_column_dict = {}
        self._multi_hash_dict = {}

        self._fg_config = fg_conf
        self._fc_config = fc_conf
        self.build_column()

    def build_column(self, **kwargs):
        for column_conf in self._fc_config['feature_columns']:
            assert 'transform_name' in column_conf, 'param transform_name not found in \n{}'.format(column_conf)
            self.insert_global_params(column_conf) # init_op:zero
            self.build_multi_hash_dict(column_conf)
            # column_conf['transform_name'] = "embedding_column" "real_valued_column" ...
            getattr(column_generator, column_conf['transform_name'])(column_conf, self._feature_column_dict, **kwargs)

    def get_column(self, name):
        assert name in self._feature_column_dict, 'feature {} does not exist'.format(name)
        return self._feature_column_dict[name]

    def get_column_list(self, name_list):
        column_list = []
        for name in name_list:
            column = self._feature_column_dict[name]
            if isinstance(column, list):
                column_list.extend(column)
            else:
                column_list.append(column)
        return column_list

    def insert_global_params(self, column_conf):
        params = ['init_op']
        for k in params:
            v = getattr(self._FLAGS, k, None)
            if v is not None and k not in column_conf:
                column_conf[k] = v

    def build_multi_hash_dict(self, column_conf):
        if 'sequence_column' == column_conf['transform_name']:
            sequence_name = column_conf['sequence_name']
            for att_fea in column_conf['features']:
                att_seq_name = '{}_{}'.format(sequence_name, att_fea['feature_name'])
                if 'hash_bucket_size' in att_fea and isinstance(att_fea['hash_bucket_size'], list):
                    self._multi_hash_dict[att_seq_name] = len(att_fea['hash_bucket_size'])
        elif 'hash_bucket_size' in column_conf and isinstance(column_conf['hash_bucket_size'], list):
            self._multi_hash_dict[column_conf['feature_name']] = len(column_conf['hash_bucket_size'])


    # get embedding according to different hash_bucket_size with different feature name
    def update_multi_hash_features(self, fg_features):
        if not self._multi_hash_dict:
            return None
        for feature_name, hash_cnt in self._multi_hash_dict.items():
            tensor = fg_features.get(feature_name, None)
            if tensor is not None:
                for i in range(hash_cnt):
                    fg_features.update({'{}_{}'.format(feature_name, i): tensor})


