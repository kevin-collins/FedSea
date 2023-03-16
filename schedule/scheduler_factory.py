import os
import importlib

current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(current_dir)


class SchedulerFactory(object):

    def __init__(self):
        pass

    @staticmethod
    def get_scheduler(class_path, *args, **kwargs):
        module_name, class_name = class_path.strip().rsplit('.', 1)
        scheduler_module = importlib.import_module(module_name)
        scheduler_instance = getattr(scheduler_module, class_name)(*args, **kwargs)
        return scheduler_instance