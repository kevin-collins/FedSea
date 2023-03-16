import os
import importlib

current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(current_dir)


class DatasetFactory(object):

    def __init__(self):
        pass

    @staticmethod
    def get_dataset(class_path, *args, **kwargs):
        module_name, class_name = class_path.strip().rsplit('.', 1)
        dataset_module = importlib.import_module(module_name)
        dataset_instance = getattr(dataset_module, class_name)(*args, **kwargs)
        return dataset_instance
