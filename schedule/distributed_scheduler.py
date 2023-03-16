# -*- coding: utf-8 -*-
from common.base.scheduler import BaseScheduler
from dataset.dataset_factory import DatasetFactory
from common.base.model import BaseModel
from dataset.csv_dataset import get_dataset
from model.model_factory import ModelFactory
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np
import time
from utils.util import get_session

class DistributedScheduler(BaseScheduler):

    def __init__(self, FLAGS, *args, **kwargs):
        self.FLAGS = FLAGS
        ps_spc = FLAGS.ps_hosts.split(",")
        worker_spc = FLAGS.worker_hosts.split(",")
        self.cluster = tf.train.ClusterSpec({"ps": ps_spc, "worker": worker_spc})

        super(DistributedScheduler, self).__init__(*args, **kwargs)

    def run(self):
        server = tf.train.Server(self.cluster,
                                 job_name=self.FLAGS.job_name,
                                 task_index=self.FLAGS.task_index,
                                 config=BaseModel.get_schedule_config(),
                                 protocol='grpc++')
        if self.FLAGS.job_name == 'ps':
            server.join()
        elif self.FLAGS.job_name == 'worker':
            is_train = (self.FLAGS.mode == 'train')
            is_chief = (self.FLAGS.task_index == 0)
            with tf.device(
                    tf.train.replica_device_setter(worker_device='/job:worker/task:{}'.format(self.FLAGS.task_index),
                                                   cluster=self.cluster)):
                train_batch_datas, test_batch_datas = get_dataset(self.FLAGS.train_files, self.FLAGS.test_files, self.FLAGS.batch_size, self.FLAGS.num_epochs if is_train else 1)

            print(self.FLAGS.model_name)
            with tf.device(
                    tf.train.replica_device_setter(worker_device='/job:worker/task:{}'.format(self.FLAGS.task_index),
                                                   cluster=self.cluster,
                                                   ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                                                       len(self.FLAGS.ps_hosts.split(',')),
                                                       tf.contrib.training.byte_size_load_fn))):
                model = ModelFactory.get_model(self.FLAGS.model_name, self.FLAGS)

                if is_train and self.FLAGS.task_index!=0:
                    model.build(train_batch_datas)
                else:
                    model.build(test_batch_datas)

                self.stop_flag = tf.Variable(initial_value=0, name='stop_flag', trainable=False,
                                             collections=[tf.GraphKeys.GLOBAL_VARIABLES])

                if self.FLAGS.same_init:
                    self.avg_step = tf.Variable(initial_value=-1, name='avg_step', trainable=False,
                                                collections=[tf.GraphKeys.GLOBAL_VARIABLES])
                else:
                    self.avg_step = tf.Variable(initial_value=0, name='avg_step', trainable=False,
                                                collections=[tf.GraphKeys.GLOBAL_VARIABLES])

                self.avg_step_add = tf.assign_add(self.avg_step, 1, use_locking=True)

                self.index = tf.Variable(initial_value=range(self.FLAGS.country_nums), name='index', trainable=False,
                                         collections=[tf.GraphKeys.GLOBAL_VARIABLES])

                self.now_round = 0
                self.start_time = time.time()

                self.gan_flag = tf.Variable(initial_value=0, name='gan_flag', trainable=False,
                                            collections=[tf.GraphKeys.GLOBAL_VARIABLES])
                self.gan_step_add = tf.assign_add(self.gan_flag, 1, use_locking=True)



                print("is_trian {}, is_chief {}".format(is_train,is_chief))

                saver = tf.train.Saver()
                print("start init...")
                with tf.train.MonitoredTrainingSession(master=server.target,
                                                       is_chief=is_chief,
                                                       checkpoint_dir=self.FLAGS.ckpt_path,
                                                       save_checkpoint_secs=None,
                                                       save_summaries_steps=None,
                                                       save_summaries_secs=None,
                                                       scaffold=model.get_scaffold(),
                                                       config=model.get_schedule_config(),
                                                       hooks=model.get_hooks(),
                                                       chief_only_hooks=model.get_chief_only_hooks()
                                                       ) as mon_sess:
                    try:
                        if not mon_sess.should_stop():
                            model.runner.before_run(mon_sess)
                    except tf.errors.OutOfRangeError:
                        print('Run out of data.')
                    print('init done!')
                    print("round {}/{} start...".format(self.now_round,self.FLAGS.round))
                    self.start_time = time.time()


                    try:
                        if is_chief:
                            self.stop_flag.load(0, mon_sess)
                            print("work0 start!")
                            while not mon_sess.should_stop():
                                if is_train and self.FLAGS.agg and self.FLAGS.same_init and mon_sess.run(self.avg_step)==-1:
                                    print("start same init...")
                                    global_vars = mon_sess.run(tf.trainable_variables(scope='Global_SubModel'))

                                    for i in range(len(self.FLAGS.countrys)):
                                        t1 = time.time()
                                        for variable, value in zip(tf.trainable_variables(
                                                scope='{}_SubModel'.format(self.FLAGS.countrys[i])),
                                                global_vars):
                                            variable.load(value, mon_sess)
                                        print("{} is done,time is {}".format(self.FLAGS.countrys[i], time.time() - t1))

                                    print("same init done!")
                                    self.avg_step.load(0, mon_sess)

                                # print("start eval")
                                if model.runner.run(mon_sess):
                                    print("chief out")
                                    break

                                now_avg_step = mon_sess.run(self.avg_step)
                                if is_train and self.FLAGS.agg and now_avg_step>self.FLAGS.avg_freq:

                                    print('start avg!')

                                    sum_vars = None
                                    var_nums = None

                                    sum_weight = 0.0

                                    # get sum vars
                                    index_now = mon_sess.run(self.index)

                                    print("start calculate avg vars...")
                                    for i in index_now:
                                        # local_vars = mon_sess.run([v for v in tf.trainable_variables(
                                        #     scope='{}_SubModel'.format(self.FLAGS.countrys[i])) if not any(key in v.name for key in self.FLAGS.avg_scope)])
                                        local_vars = mon_sess.run([v for v in tf.trainable_variables(
                                            scope='{}_SubModel'.format(self.FLAGS.countrys[i])) if not any(key in v.name for key in self.FLAGS.avg_scope)])
                                        # print(local_vars[0][0])
                                        print("last var_nums is {}, now {} var_nums is {}".format(var_nums, self.FLAGS.countrys[i], len(local_vars)))

                                        # make sure len is equal
                                        if var_nums is None:
                                            var_nums = len(local_vars)
                                        else:
                                            assert var_nums==len(local_vars)


                                        local_weight = 1.
                                        sum_weight += local_weight
                                        # print('sample_nums is {},so add weight is {}'.format(local_weight, local_weight / float(sum_weight)))
                                        print('{} sample_nums is {}'.format(self.FLAGS.countrys[i],local_weight))
                                        if sum_vars is None:
                                            sum_vars = local_vars
                                            for i in range(len(sum_vars)):
                                                sum_vars[i] = local_vars[i] * local_weight
                                        else:
                                            for i in range(len(sum_vars)):
                                                sum_vars[i] += local_vars[i] * local_weight

                                    print('sum sample_num is {}'.format(sum_weight))
                                    # get avg vars
                                    global_vars = []
                                    for var in sum_vars:
                                        global_vars.append(var / float(sum_weight))
                                    print("avg vars calculate done!")

                                    for i in range(len(self.FLAGS.countrys)):
                                        t1 = time.time()
                                        for variable, value in zip([v for v in tf.trainable_variables(
                                        scope='{}_SubModel'.format(self.FLAGS.countrys[i])) if not any(key in v.name for key in self.FLAGS.avg_scope)], global_vars):
                                            variable.load(value,mon_sess)
                                        print("{} is done,time is {}".format(self.FLAGS.countrys[i], time.time()-t1))
                                    saver.save(get_session(mon_sess), self.FLAGS.ckpt_path + "/" + str(self.now_round) + "_model.ckpt")
                                    print('avg done!')

                                if is_train and self.FLAGS.FedSFA and now_avg_step > self.FLAGS.avg_freq:
                                    self.gan_flag.load(1, mon_sess)
                                    print("work0 signal to gan process!")
                                    while mon_sess.run(self.gan_flag) >= 1:
                                        print("work0 wait train gan round finish, now gan step {}/{}...".format(
                                            mon_sess.run(self.gan_flag), self.FLAGS.gan_map_step))
                                        time.sleep(5)

                                if is_train and now_avg_step > self.FLAGS.avg_freq:
                                    # set round
                                    print("round {}/{} end!  time:{}s".format(self.now_round, self.FLAGS.round,time.time()-self.start_time))
                                    self.start_time = time.time()
                                    self.now_round += 1

                                    if is_train and self.now_round >= self.FLAGS.round:
                                        # sign finish
                                        self.stop_flag.load(1, mon_sess)
                                        print("round meets the requirement, chief out")
                                        break

                                    # load new index
                                    index_new = np.sort(
                                        np.random.choice(len(self.FLAGS.countrys), self.FLAGS.country_nums,
                                                         replace=False))

                                    self.index.load(index_new, mon_sess)
                                    print("get new index", index_new)

                                    # sign restart
                                    self.avg_step.load(0, mon_sess)

                                    print("round {}/{} start...".format(self.now_round, self.FLAGS.round))

                        else:
                            print("work start!")
                            while not mon_sess.should_stop():
                                if is_train and mon_sess.run(self.stop_flag)==1:
                                    print("chief 0 is out,so work out!")
                                    break

                                while is_train and self.FLAGS.agg and self.FLAGS.same_init and mon_sess.run(self.avg_step)==-1:
                                    print("{} is waiting for same init done".format(self.FLAGS.task_index))
                                    time.sleep(5)
                                # run each batch
                                if model.runner.run(mon_sess):
                                    break

                                # agg
                                if is_train:

                                    if self.FLAGS.agg:

                                        # if not should stop,work 0 not out,is_train,is_agg
                                        while not mon_sess.should_stop() and is_train and self.FLAGS.agg and mon_sess.run(self.stop_flag)!=1 and mon_sess.run(self.gan_flag)==0\
                                                and mon_sess.run(self.avg_step)>self.FLAGS.avg_freq:
                                            print("{} is waiting for avg done".format(self.FLAGS.task_index))
                                            time.sleep(5)

                                        if self.FLAGS.FedSFA:
                                            # print("gan_step {}".format(mon_sess.run(self.gan_flag)))
                                            while not mon_sess.should_stop() and is_train and self.FLAGS.agg and mon_sess.run(
                                                    self.stop_flag) != 1 and mon_sess.run(
                                                self.avg_step) > self.FLAGS.avg_freq \
                                                    and mon_sess.run(self.gan_flag) < 1:
                                                print("wait signal to start gan process...")
                                                time.sleep(3)

                                            if mon_sess.run(self.gan_flag) >= self.FLAGS.gan_map_step:
                                                self.gan_flag.load(0, mon_sess)
                                                print("gan process is done")
                                            elif mon_sess.run(self.gan_flag) >= 1:
                                                mon_sess.run(self.gan_step_add)
                                                # print("work train gan round now")
                                                continue

                                        while not mon_sess.should_stop() and is_train and mon_sess.run(self.stop_flag)!=1 and mon_sess.run(self.gan_flag)==0\
                                                and mon_sess.run(self.avg_step)>self.FLAGS.avg_freq:
                                            print("{} is waiting for generate new index".format(self.FLAGS.task_index))
                                            time.sleep(1)



                                # # add avg step
                                mon_sess.run(self.avg_step_add)
                    # break
                    except tf.errors.OutOfRangeError:
                        print('Run out of data.')
                    finally:
                        if is_chief and is_train:
                            self.stop_flag.load(1, mon_sess)
                            print("already require other work to stop!")
                        if not mon_sess.should_stop():
                            model.runner.after_run(mon_sess)


