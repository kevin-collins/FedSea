import os
os.environ['TF_JIT_PROFILING'] = 'False'
os.environ['OPTIMIZE_START_TIME'] = 'True'
import sys
import random
import tensorflow as tf
import numpy as np
random.seed(0)
tf.set_random_seed(0)
np.random.seed(0)
current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(current_dir)

sys.path.append(root_dir)

from schedule.scheduler_factory import SchedulerFactory
from utils.config import parse_global_conf


flags = tf.app.flags
flags.DEFINE_string('global_conf', "./conf/param_fed.json", 'global parameters')

flags.DEFINE_integer('task_index', 0, 'worker task index')
flags.DEFINE_string('job_name', '', 'worker/ps')
flags.DEFINE_string('ps_hosts', "127.0.0.1:1233", 'ps hosts')
flags.DEFINE_string('worker_hosts', "127.0.0.1:1234", 'worker hosts')


FLAGS = tf.app.flags.FLAGS
# flags.global_conf = "./conf/param_fed.json"
parse_global_conf(FLAGS)

g = os.walk("./")
for dirpath, dirnames, filenames in g:
    print(dirnames,filenames)

print()
g1 = os.walk(FLAGS.train_files)
g2 = os.walk(FLAGS.test_files)
FLAGS.train_files = []
FLAGS.test_files = []
for dirpath, dirnames, filenames in g1:
    for filepath in filenames:
        if filepath.endswith('csv') and '._' not in filepath:
            FLAGS.train_files.append(os.path.join(dirpath, filepath))

for dirpath, dirnames, filenames in g2:
    for filepath in filenames:
        if filepath.endswith('csv') and '._' not in filepath:
            FLAGS.test_files.append(os.path.join(dirpath, filepath))
print(FLAGS.train_files, FLAGS.test_files)

assert len(FLAGS.train_files)==len(FLAGS.test_files)==len(FLAGS.countrys)

def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)

    scheduler = SchedulerFactory.get_scheduler(FLAGS.scheduler_name, FLAGS)
    scheduler.run()


if __name__ == '__main__':
    tf.app.run()
