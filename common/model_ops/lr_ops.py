import tensorflow as tf
import numpy as np
import math

def lr_warm_restart(lr_max, globalstep, lr_min, periodic_step):
  """
  From Paper "SGDR: Stochastic Gradient Descent with Warm Restarts"
  :param lr_max:
  :param globalstep:
  :param lr_min:
  :param periodic_step:
  :return: Tensor represents new learning rate
  """
  lr_min = tf.convert_to_tensor(lr_min)
  periodic_step = tf.convert_to_tensor(periodic_step, dtype=tf.int32)
  PI = tf.convert_to_tensor(math.pi)
  tlr = (lr_max + lr_min + (lr_max - lr_min) * tf.cos(
    PI * tf.cast(tf.mod(globalstep, periodic_step),tf.float32) / tf.cast(periodic_step,tf.float32))) * 0.5
  return tlr

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
  """Cosine decay schedule with warm up period.
    global_step: int64 (scalar) tensor representing global step.
    learning_rate_base: base learning rate.
    total_steps: total number of training steps.
    warmup_learning_rate: initial learning rate for warm up.
    warmup_steps: number of warmup steps.
    hold_base_rate_steps: Optional number of steps to hold base learning rate
      before decaying.
  """
  if total_steps < warmup_steps:
    raise ValueError('total_steps must be larger or equal to '
                     'warmup_steps.')
  def eager_decay_rate():
    """Callable to compute the learning rate."""
    learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
        np.pi *
        (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
        ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
      learning_rate = tf.where(
          global_step > warmup_steps + hold_base_rate_steps,
          learning_rate, learning_rate_base)
    if warmup_steps > 0:
      if learning_rate_base < warmup_learning_rate:
        raise ValueError('learning_rate_base must be larger or equal to '
                         'warmup_learning_rate.')
      slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
      warmup_rate = slope * tf.cast(global_step,
                                    tf.float32) + warmup_learning_rate
      learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                               learning_rate)
    return tf.where(global_step > total_steps, 0.0, learning_rate,
                    name='learning_rate')

  return eager_decay_rate()