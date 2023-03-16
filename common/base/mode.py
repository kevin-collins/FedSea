class ModeKeys(object):
    """Standard names for model modes.
    The following standard keys are defined:
    * `TRAIN`: training/fitting mode.
    * `EVAL`: testing/evaluation mode.
    * `PREDICT`: predication/inference mode.
    """

    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'predict'
    LOCAL = 'local'
    EXPORT = 'export'
    DEBUG = 'debug'
    ANALYSIS = 'analysis'
    GLOBAL = 'global'