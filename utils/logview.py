import time
import re

def __is_start_training(log):
    # train_tuned: running
    # train_tuned: 2019-11-09 20:22:39 ps_job:2/0/2[0%]	worker_job:0/0/10[0%]
    if log.strip().startswith('train_tuned: ') and 'ps_job' in log:
        return True
    return False


def format_experiment_logview(logfile, data_sha):
    """ 监控日志文件，处理pai的输出，格式化logview.
    - ph
    >**project**: xxxx       **word id**: 20191021114525740g7e2pfdv2
    >**logview**: [link](xxxxxxx),  **start**: xxxxxxx


    每1秒读取一次文件， 知道第一行运行日志输： train_tuned: 2019-11-09 20:48:59 ps_job:2/0/2[0%]	worker_job:6/0/10[0%]
    """
    times = 0
    find = False
    with open(logfile, 'r') as fp:
        while True:
            fp.seek(0)
            lines = fp.readlines()
            if len(lines) >= 5:
                for i in range(len(lines)):
                    if __is_start_training(lines[i]):
                        find=True
                        break
                if find:
                    break
            time.sleep(1)
            times += 1
            if times > 600:
                print('Parse log file faild.')
                exit()
    # parse
    command, work_id, instance_id, logview = lines[:4]
    first_log = lines[i]

    # parse project  <-- odps://product_dump_dev/tables
    project = re.findall('odps://\D*/tables', command)[0][7:-7]
    
    # parse work_id  <-- ID = 20191109125113737gu3yqdep2
    work_id = work_id.split('=')[1].strip()

    # parse logview
    logview = logview.strip('\n')

    # parse experiment configures
    experiment, version, nation = command.split('--env_str=')[-1].split(',')[:3]
    oss_path =  'oss://deeprank/bigmodel/{}/checkpoints/{}/{}'.format(nation, experiment, version)


    # parse start time  <-- train_tuned: 2019-11-09 20:51:37 ps_job:2/0/2[0%]	worker_job:3/0/10[0%]
    start = first_log.strip().split('train_tuned: ')[1].strip().split(' ps_job')[0].strip()

    format =  '**project** : %s      **worker id** : %s\n'
    format += '**logview** :[link](%s)      **start** : %s      **data** : ` %s `\n'
    format += "**logfile** : ` logviews%s ` \n"  % logfile.split('logviews')[1].strip()
    format += "**oss** : ` %s ` "  % oss_path

    print('\nExperiment Logview here, copy it to markdown !!!\n\n', format % (project, work_id, logview, start, data_sha), '\n')


if __name__ == "__main__":
    import sys
    format_experiment_logview(sys.argv[1], sys.argv[2])
