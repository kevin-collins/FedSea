import sys
import hashlib
import time


def run(args):
    exp_name=args[1]
    detail=args[2]
    tn=args[3]
    tables = args[4]
    oss = args[5]

    hash_n = 8
    tables = tables.split(',')
    data = str(tables)
    sha = hashlib.sha1(data.encode('utf-8')).hexdigest()[:hash_n]
    bizdate = tables[1].split('ds')[-1]

    with open('/tmp/bigm_data_sha', 'w') as f:
        f.write(sha)
    print('{}: {}'.format(sha, data))
    oss = oss + '_{}_{}'.format(sha,tn)

    print('\nCopy this Log:')
    print('**start**:{}\n**version**:{}/tn={}\n**comment**:{}\n**data_sha**:{}\n**oss**:{}\n**tables**:{}\n\n'.format(
        time.strftime('%Y-%m-%d %H:%M'),
        exp_name,
        tn,
        detail,
        sha,
        oss,
        data
    ))
    with open('/tmp/bigm_data_sha', 'w') as f:
        f.write(sha)

if __name__ == '__main__':
    print(sys.argv)
    run(sys.argv)