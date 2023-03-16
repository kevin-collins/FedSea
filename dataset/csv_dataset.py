# -*- coding: utf-8 -*-
import tensorflow as tf

def CSV_read(filelist,batch_size=512, num_epochs=1):

    queue = tf.train.string_input_producer(filelist,num_epochs=num_epochs)


    reader = tf.TextLineReader()
    _, value = reader.read_up_to(queue, batch_size)

    record_defaults = [[""], [""], ["0"], ["0"]]
    sample = tf.decode_csv(value,field_delim=",",record_defaults=record_defaults)
    # SBT sign data remaining in queue proportion now
    batch_data = tf.train.shuffle_batch_join([sample],
                                             batch_size=batch_size,
                                             capacity=batch_size * 5,
                                             enqueue_many=True,
                                             min_after_dequeue=batch_size
                                             )
    return batch_data
def get_dataset(train_files,test_files,batch_size=512, num_epochs=1):
    train_ds = []
    test_ds = []
    for f in train_files:
        if f.endswith('csv'):
            batch_data = CSV_read([f],batch_size,num_epochs)
            train_ds.append(batch_data)

    for f in test_files:
        if f.endswith('csv'):
            batch_data = CSV_read([f],batch_size,num_epochs)
            test_ds.append(batch_data)
    return train_ds,test_ds
