import tensorflow as tf


def safe_log(x):
    return tf.log(tf.clip_by_value(x, 1e-8, 1.0))


def bce(y_true, y_pred, mask=None):
    pt_1 = y_true * safe_log(y_pred)
    pt_0 = (1-y_true) * safe_log(1-y_pred)
    if mask is not None:
        return -tf.reduce_sum((pt_0 + pt_1)*mask) / (tf.reduce_sum(mask) + 1e-8)
    return -tf.reduce_mean(pt_0 + pt_1)



def focal_loss(gamma=2, alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):#with tensorflow
        eps = 1e-12
        y_pred=tf.clip_by_value(y_pred,eps,1.-eps)  # improve the stability of the focal loss and see issues 1 for more information
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1))-tf.reduce_sum((1-alpha) * tf.pow( pt_0, gamma) * tf.log(1. - pt_0))
    return focal_loss_fixed


def sparse_cross_entropy_with_logits(y_true, y_pred, mask=None):
    with tf.name_scope("sparse_cross_entropy_with_logits"):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred+1e-10,labels=tf.cast(y_true, tf.int64))
        cost = tf.reduce_mean(loss)
    return cost


class LossFunc():
    def __init__(self, name):
        #super(LossFunc, self).__init__('Loss Function')
        self.name = name

        self.losses = {
            'bce': bce,
            # 'ghm': GHM_Loss(momentum=0.75).ghm_class_loss,
            'sparse_cross_entropy_with_logits': sparse_cross_entropy_with_logits,
        }

    def get_loss(self, name):
        if name not in self.losses:
            print('Loss: `{}` is not supported. {}'.format(
                name, str(list(self.losses.keys()))
            ))
            exit()

        return self.losses[name]

    def __call__(self, y_true, y_pred, *args, **kwargs):
        return self.get_loss(self.name)(y_true, y_pred, *args, **kwargs)





# =-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-=


def mutli_task_learning_loss_with_uncertainty_paper(loss_list={}):
    assert len(loss_list) > 0

    losses = []
    sigma, factor, bias = {}, {}, {}

    for name, loss in loss_list.items():
        log_sigma_2  = tf.get_variable(
            name   = 'uncertainty_%s' % name,
            dtype  = tf.float32,
            shape  = [],
            initializer=tf.initializers.random_uniform(minval=-1, maxval=0))


        sigma[name]  = tf.sqrt(tf.exp(log_sigma_2))
        factor[name] = tf.div(1.0, tf.multiply(2.0, tf.exp(log_sigma_2)))
        bias[name]   = tf.log(sigma[name])

        losses.append(tf.add(tf.multiply(factor[name], loss), bias[name]))

    return tf.reduce_sum(losses),  {'sigma': sigma, 'factor': factor, 'bias': bias}


def mutli_task_learning_loss_with_uncertainty(loss_list={}):
    assert len(loss_list) > 0
    sigmas, factors, biases = {}, {}, {}
    losses = []
    for name, loss in loss_list.items():
        sigma  = tf.get_variable(
            name   = 'uncertainty_%s' % name,
            dtype  = tf.float32,
            shape  = [],
            initializer=tf.initializers.random_uniform(minval=0.2, maxval=1))

        factor = tf.div(1.0, tf.multiply(2.0, sigma))

        sigmas[name]  = sigma
        factors[name] = factor
        biases[name]  = tf.log(sigma)

        losses.append(tf.add(tf.multiply(factor, loss), biases[name]))

    return tf.reduce_sum(losses),  {'sigma': sigmas, 'factor': factors, 'bias': biases}

class MultiLossLayer():
    def __init__(self, name):
        self.name = name
        self.multi_loss_hanndles = {
            'sum': self.sum,
            'avg': self.avg,
            'weighted': self.weighted,
            'uncertainty': mutli_task_learning_loss_with_uncertainty
        }

    def sum(self, loss_dict):
        return tf.reduce_sum(loss_dict.values())

    def avg(self, loss_dict):
        return tf.reduce_mean(loss_dict.values())

    def weighted(self, loss_dict, weight_dict):
        pass


    def get_hanndle(self, name):
        if name not in self.multi_loss_hanndles:
            print('Hanndle: `{}` is not supported. {}'.format(
                name, str(list(self.multi_loss_hanndles.keys()))
            ))
            exit()

        return self.multi_loss_hanndles[name]

    def __call__(self, loss_dict, *args, **kwargs):
        return self.get_hanndle(self.name)(loss_dict, *args, **kwargs)
