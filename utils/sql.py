import subprocess
import collections
import subprocess
import os


class SQLBase(object):
    def __init__(self):
        pass


class SQLScript(SQLBase):
    def __init__(self, odps_cmd, odps_conf):
        super(SQLScript, self).__init__()
        self.odps_cmd = odps_cmd
        self.odps_conf = odps_conf

    def call(self, sql, to_file):
        if to_file:
            cmd = "%s --config='%s' -e \"%s\" > %s" % (self.odps_cmd, self.odps_conf, sql, to_file)
        else:
            cmd = "%s --config='%s' -e \"%s\"" % (self.odps_cmd, self.odps_conf, sql)
        subprocess.Popen(cmd, shell=True).wait()

    def parse_mapping(self, mapping_string):
        mapping = collections.OrderedDict()
        for m in mapping_string.strip().replace('\'', '').split('\n'):
            if m.startswith('--'):
                continue
            old, new = m.strip().split(':')
            mapping[old.strip()] = new.strip()

        return mapping

    def run(self, sql=None, to_file=None, args={}):
        if sql.endswith('.sql'):
            # input a sql as a txt file
            with open(sql, 'r') as f:
                sql = f.read().strip()
        else:
            # input a sql with string format
            sql = sql

        # get mapping
        if '-- mapping to markdown --' in sql:
            sql, mapping = sql.split('-- mapping to markdown --')
            mapping = self.parse_mapping(mapping)

        # replace args
        for old, new in args.items():
            sql = sql.replace(old, new)

        self.call(sql, to_file=to_file)


def aligned_table_from_sql_output(output):
    with open(output, 'r') as f:
        context = f.read()

    auc = 0.0
    n = 0
    output = context.replace('-', '').replace('+', '')
    lines = []
    max_len = {}
    for line in output.split('\n'):
        if line:
            items = []
            for inx, item in enumerate(line.strip('|').split('|')):
                item = ' ' + item.strip() + ' '
                char_cnt = len(item)
                items.append(item)
                if inx == 1:
                    try:
                        auc += float(item)
                        n += 1
                    except:
                        auc += 0
                max_len[str(inx)] = char_cnt if max_len.get(str(inx), 0) < char_cnt else max_len.get(str(inx), 0)
            lines.append(items)

    side = '+' + '+'.join(['-' * cnt for cnt in max_len.values()]) + '+'
    format = '|' + '|'.join(['{:^%ds}' % cnt for cnt in max_len.values()]) + '|'
    print(side)
    for line in lines:
        print(format.format(*line))
        print(side)
    print('avg auc',auc/n)

# if __name__ == '__main__':
#     # sql_script = SQLScript(odps_cmd = '/Users/codewang/bins/odpscmd/bin/odpscmd',
#     #                        odps_conf= '/Users/codewang/bins/odpscmd/conf/odps_config.ini.product_dump')
#     # sql_script.run(sql = './sql_hub/compute_metrics_v2.sql',
#     #                to_file='./.output',
#     #                args= {
#     #                   r'${experiment}' : 'esmm_online_ltr',
#     #                   r'${version}': 'v2.0.8',
#     #                   r'${nation}' : 'VN',
#     #                   r'${data}':  '9074f9d'
#     #               })
#
#     aligned_table_from_sql_output('../logs/metrics/fedavg_part_feature_local_map_user_28.output')