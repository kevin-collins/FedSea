from abc import ABCMeta, abstractmethod


class BaseDataset(object):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_batch(self, *args, **kwargs):
        raise NotImplementedError('Calling an abstract method.')
