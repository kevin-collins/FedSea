from abc import ABCMeta, abstractmethod


class BaseScheduler(object):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError('Calling an abstract method.')

