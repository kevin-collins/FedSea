import os
import sys
import importlib

current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(current_dir)


class ModelFactory(object):

    def __init__(self):
        pass

    @staticmethod
    def get_model(class_path, *args, **kwargs):
        module_name, class_name = class_path.strip().rsplit('.', 1)
        model_module = importlib.import_module(module_name)
        model_instance = getattr(model_module, class_name)(*args, **kwargs)
        return model_instance
