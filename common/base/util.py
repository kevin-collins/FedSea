import time

fmt = '%Y%m%d %H:%M:%S'


def logger(*message):
    print('{} -> {}'.format(time.strftime(str(fmt)), ' '.join([str(i) for i in message])))