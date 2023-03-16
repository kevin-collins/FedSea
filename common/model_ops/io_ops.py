#coding:utf8
import tensorflow as tf

def global_string_input_producer(tables,
                                 sliceId,
                                 cluster,
                                 is_chief,
                                 batch_size,
                                 num_epoch,
                                 shuffle,
                                 selected_cols,
                                 capacity,
                                 record_defaults,
                                 name = None,
                                 num_threads=4,
                                 partition_size=65536,
                                 csv_delimiter='\t',
                                 enqueue_many=True,
                                 log_inputs=True):

    '''
    :param tables: input tables
    :param sliceId: worker index
    :param cluster: cluster
    :param is_chief: is chief worker
    :param batch_size:
    :param num_epoch:
    :param shuffle: is shuffle? True or False
    :param selected_cols: column name. e.g. 'unique_id,features,weights'
    :param capacity: e.g. 16 * batch_size
    :param record_defaults: default values. e.g.[[''],[''],[1.0]]
    :param name:
    :param num_threads:
    :param partition_size:
    :param csv_delimiter:
    :param enqueue_many:
    :param log_inputs:
    :return:
    '''
    # global_input_queue放在ps上
    availableWorkerDevice = "/job:worker/task:%d" % sliceId
    with tf.device(tf.train.replica_device_setter(worker_device=availableWorkerDevice, cluster=cluster)):
        splits = tf.contrib.global_input.partition_filenames_once(tables,partition_size=partition_size)
        input_queue = tf.contrib.global_input.global_string_input_producer(string_tensor=splits,
                                                                           num_epochs=num_epoch,
                                                                           shuffle=shuffle,
                                                                           is_chief=is_chief,
                                                                           log_inputs=log_inputs,
                                                                           name=name,
                                                                           seed=batch_size)

    # io还是放在local device上的
    workerDevice = "/job:worker/task:%d/cpu:%d" % (sliceId, 0)
    with tf.device(workerDevice):

        reader = tf.TableRecordReader(csv_delimiter=csv_delimiter,
                                      selected_cols=selected_cols,
                                      num_threads=num_threads,
                                      capacity=capacity)
        _, values = reader.read_up_to(input_queue, num_records=batch_size)
        batch_values = tf.train.batch([values],
                                      batch_size=batch_size,
                                      capacity=capacity,
                                      enqueue_many=enqueue_many,
                                      num_threads=num_threads)
        features = tf.decode_csv(batch_values,
                                 record_defaults=record_defaults,
                                 field_delim=csv_delimiter)
        return features