import hashlib
import os
import json

class TableRecord(object):
    def __init__(self, context):

        self.context = json.loads(context)
        train_table = self.context['data']['train_data']
        test_table  = self.context['data']['test_data']

        self.hash_n = 7
        self.data = 'train:%s test:%s' % (train_table, test_table)
        self.local_record = 'utils/.odps_tables.record'
        self.sha = hashlib.sha1(self.data.encode('utf-8')).hexdigest()[:self.hash_n]
        self.localize()
        

    def localize(self):
        # check if current data record is allready exists in record file.
        records = self.load()
        if self.sha in records[0]:
            index = records[0].index(self.sha)
            if records[1][index] != self.data:
                print('same SHA1(%s) code between different data' % self.hash_n)
                print('now:        %s,  %s' % (self.sha, self.data))
                print('in record:  %s,  %s' % (records[0][index], records[1][index]))
                exit()
            else:
                pass  # do nothing
        else:
            self.write(self.sha, self.data)



    def load(self):
        records = [[], []]
        if os.path.exists(self.local_record):
            with open(self.local_record, 'r') as fp:
                for line in fp:
                    _parts = line.strip().split(' ')
                    sha = _parts[0]
                    data_records = ' '.join(_parts[1:])
                    records[0].append(sha)
                    records[1].append(data_records)
        return records
    
    def write(self, sha, data):
        with open(self.local_record, 'a') as fp:
            fp.write('%s %s\n' % (sha, data))
    
