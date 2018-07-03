import tensorflow as tf
import numpy as np
import time
import read_TF_records
from tensorflow.python.training.moving_averages import assign_moving_average
import cv2
import copy

with tf.variable_scope('p_conv_par'):
    conv_p_par_weight = None
    conv_p_par_bias = None
    conv_p_par_scale_par = None
    conv_p_par_offset_par = None
    conv_p_par_me_EMA = None
    conv_p_par_va_EMA = None

with tf.variable_scope('r_conv_par'):
    conv_r_par_weight = None
    conv_r_par_bias = None
    conv_r_par_scale_par = None
    conv_r_par_offset_par = None
    conv_r_par_me_EMA = None
    conv_r_par_va_EMA = None

with tf.variable_scope('o_conv_par'):
    conv_o_par_weight = None
    conv_o_par_bias = None
    conv_o_par_scale_par = None
    conv_o_par_offset_par = None
    conv_o_par_me_EMA = None
    conv_o_par_va_EMA = None


def init_par_p():
    global conv_p_par_weight, conv_p_par_bias, conv_p_par_scale_par, conv_p_par_offset_par, conv_p_par_me_EMA, conv_p_par_va_EMA
    with tf.variable_scope('p_conv_par'):
        conv_p_par_weight = {'conv_p_1': tf.Variable(tf.random_normal([3, 3, 3, 10]), name='conv_p_1'),
                             'conv_p_2': tf.Variable(tf.random_normal([3, 3, 10, 16]), name='conv_p_2'),
                             'conv_p_3': tf.Variable(tf.random_normal([3, 3, 16, 32]), name='conv3'),
                             'conv_p_cls': tf.Variable(tf.random_normal([1, 1, 32, 2]), name='conv_p_cls'),
                             'conv_p_reg': tf.Variable(tf.random_normal([1, 1, 32, 4]), name='conv_p_reg')}
        conv_p_par_bias = {'bias_p_1': tf.Variable(tf.random_normal([10]), name='bias_p_1'),
                           'bias_p_2': tf.Variable(tf.random_normal([16]), name='bias_p_2'),
                           'bias_p_3': tf.Variable(tf.random_normal([32]), name='bias_p_3'),
                           'bias_p_cls': tf.Variable(tf.random_normal([2]), name='bias_p_cls'),
                           'bias_p_reg': tf.Variable(tf.random_normal([4]), name='bias_p_reg')}
        conv_p_par_scale_par = {'bns_p_1': tf.Variable(tf.ones([10]), name='bns_p_1'),
                                'bns_p_2': tf.Variable(tf.ones([16]), name='bns_p_2'),
                                'bns_p_3': tf.Variable(tf.ones([32]), name='bns_p_3'),
                                'bns_p_cls': tf.Variable(tf.ones([2]), name='bns_p_cls'),
                                'bns_p_reg': tf.Variable(tf.ones([4]), name='bns_p_reg')}
        conv_p_par_offset_par = {'bno_p_1': tf.Variable(tf.zeros([10]), name='bno_p_1'),
                                 'bno_p_2': tf.Variable(tf.zeros([16]), name='bno_p_2'),
                                 'bno_p_3': tf.Variable(tf.zeros([32]), name='bno_p_3'),
                                 'bno_p_cls': tf.Variable(tf.zeros([2]), name='bno_p_cls'),
                                 'bno_p_reg': tf.Variable(tf.zeros([4]), name='bno_p_reg')}
        conv_p_par_me_EMA = {'conv_me_p_1': tf.Variable(tf.zeros([10]), trainable=False, name='conv_me_p_1'),
                             'conv_me_p_2': tf.Variable(tf.zeros([16]), trainable=False, name='conv_me_p_2'),
                             'conv_me_p_3': tf.Variable(tf.zeros([32]), trainable=False, name='conv_me_p_3'),
                             'conv_me_p_cls': tf.Variable(tf.zeros([2]), trainable=False, name='conv_me_p_cls'),
                             'conv_me_p_reg': tf.Variable(tf.zeros([4]), trainable=False, name='conv_me_p_reg')}
        conv_p_par_va_EMA = {'conv_va_p_1': tf.Variable(tf.ones([10]), trainable=False, name='conv_va_p_1'),
                             'conv_va_p_2': tf.Variable(tf.ones([16]), trainable=False, name='conv_va_p_2'),
                             'conv_va_p_3': tf.Variable(tf.ones([32]), trainable=False, name='conv_va_p_3'),
                             'conv_va_p_cls': tf.Variable(tf.ones([2]), trainable=False, name='conv_va_cls'),
                             'conv_va_p_reg': tf.Variable(tf.ones([4]), trainable=False, name='conv_va_reg')}
        for par in [conv_p_par_weight, conv_p_par_bias, conv_p_par_scale_par, conv_p_par_offset_par, conv_p_par_me_EMA,
                    conv_p_par_va_EMA]:
            for value in par.values():
                tf.add_to_collection('conv_p', value)


def init_par_r():
    global conv_r_par_weight, conv_r_par_bias, conv_r_par_scale_par, conv_r_par_offset_par, conv_r_par_me_EMA, conv_r_par_va_EMA
    with tf.variable_scope('r_conv_par'):
        conv_r_par_weight = {'conv_r_1': tf.Variable(tf.random_normal([3, 3, 3, 28]), name='conv_r_1'),
                             'conv_r_2': tf.Variable(tf.random_normal([3, 3, 28, 48]), name='conv_r_2'),
                             'conv_r_3': tf.Variable(tf.random_normal([2, 2, 48, 64]), name='conv_r_3'),
                             'full_r_4': tf.Variable(tf.random_normal([3 * 3 * 64, 128]), name='full_r_4'),
                             'full_r_cls': tf.Variable(tf.random_normal([128, 2]), name='full_r_cls'),
                             'full_r_reg': tf.Variable(tf.random_normal([128, 4]), name='full_r_reg')}
        conv_r_par_bias = {'bias_r_1': tf.Variable(tf.random_normal([28]), name='bias_r_1'),
                           'bias_r_2': tf.Variable(tf.random_normal([48]), name='bias_r_2'),
                           'bias_r_3': tf.Variable(tf.random_normal([64]), name='bias_r_3'),
                           'bias_r_4': tf.Variable(tf.random_normal([128]), name='bias_r_4'),
                           'bias_r_cls': tf.Variable(tf.random_normal([2]), name='bias_r_cls'),
                           'bias_r_reg': tf.Variable(tf.random_normal([4]), name='bias_r_reg')}
        conv_r_par_scale_par = {'bns_r_1': tf.Variable(tf.ones([28]), name='bns_r_1'),
                                'bns_r_2': tf.Variable(tf.ones([48]), name='bns_r_2'),
                                'bns_r_3': tf.Variable(tf.ones([64]), name='bns_r_3'),
                                'bns_r_4': tf.Variable(tf.ones([128]), name='bns_r_4'),
                                'bns_r_cls': tf.Variable(tf.ones([2]), name='bns_r_cls'),
                                'bns_r_reg': tf.Variable(tf.ones([4]), name='bns_r_reg')}
        conv_r_par_offset_par = {'bno_r_1': tf.Variable(tf.zeros([28]), name='bno_r_1'),
                                 'bno_r_2': tf.Variable(tf.zeros([48]), name='bno_r_2'),
                                 'bno_r_3': tf.Variable(tf.zeros([64]), name='bno_r_3'),
                                 'bno_r_4': tf.Variable(tf.zeros([128]), name='bno_r_4'),
                                 'bno_r_cls': tf.Variable(tf.zeros([2]), name='bno_r_cls'),
                                 'bno_r_reg': tf.Variable(tf.zeros([4]), name='bno_r_reg')}
        conv_r_par_me_EMA = {'conv_me_r_1': tf.Variable(tf.zeros([28]), trainable=False, name='conv_me_r_1'),
                             'conv_me_r_2': tf.Variable(tf.zeros([48]), trainable=False, name='conv_me_r_2'),
                             'conv_me_r_3': tf.Variable(tf.zeros([64]), trainable=False, name='conv_me_r_3'),
                             'conv_me_r_4': tf.Variable(tf.zeros([128]), trainable=False, name='conv_me_r_4'),
                             'conv_me_r_cls': tf.Variable(tf.zeros([2]), trainable=False, name='conv_me_r_cls'),
                             'conv_me_r_reg': tf.Variable(tf.zeros([4]), trainable=False, name='conv_me_r_reg')}
        conv_r_par_va_EMA = {'conv_va_r_1': tf.Variable(tf.ones([28]), trainable=False, name='conv_va_r_1'),
                             'conv_va_r_2': tf.Variable(tf.ones([48]), trainable=False, name='conv_va_r_2'),
                             'conv_va_r_3': tf.Variable(tf.ones([64]), trainable=False, name='conv_va_r_3'),
                             'conv_va_r_4': tf.Variable(tf.ones([128]), trainable=False, name='conv_va_r_4'),
                             'conv_va_r_cls': tf.Variable(tf.ones([2]), trainable=False, name='conv_va_r_cls'),
                             'conv_va_r_reg': tf.Variable(tf.ones([4]), trainable=False, name='conv_va_r_reg')}
    for par in [conv_r_par_weight, conv_r_par_bias, conv_r_par_scale_par, conv_r_par_offset_par, conv_r_par_me_EMA,
                conv_r_par_va_EMA]:
        for value in par.values():
            tf.add_to_collection('conv_r', value)


def init_par_o():
    global conv_o_par_weight, conv_o_par_bias, conv_o_par_scale_par, conv_o_par_offset_par, conv_o_par_me_EMA, conv_o_par_va_EMA
    with tf.variable_scope('o_conv_par'):
        conv_o_par_weight = {'conv_o_1': tf.Variable(tf.random_normal([3, 3, 3, 32]), name='conv_o_1'),
                             'conv_o_2': tf.Variable(tf.random_normal([3, 3, 32, 64]), name='conv_o_2'),
                             'conv_o_3': tf.Variable(tf.random_normal([3, 3, 64, 64]), name='conv_o_3'),
                             'conv_o_4': tf.Variable(tf.random_normal([2, 2, 64, 128]), name='conv_o_4'),
                             'full_o_5': tf.Variable(tf.random_normal([3 * 3 * 128, 256]), name='full_o_5'),
                             'full_o_cls': tf.Variable(tf.random_normal([256, 2]), name='full_o_cls'),
                             'full_o_reg': tf.Variable(tf.random_normal([256, 4]), name='full_o_reg'),
                             'full_o_fll': tf.Variable(tf.random_normal([256, 10]), name='full_o_fll')}
        conv_o_par_bias = {'bias_o_1': tf.Variable(tf.random_normal([32]), name='bias_o_1'),
                           'bias_o_2': tf.Variable(tf.random_normal([64]), name='bias_o_2'),
                           'bias_o_3': tf.Variable(tf.random_normal([64]), name='bias_o_3'),
                           'bias_o_4': tf.Variable(tf.random_normal([128]), name='bias_o_4'),
                           'bias_o_5': tf.Variable(tf.random_normal([256]), name='bias_o_5'),
                           'bias_o_cls': tf.Variable(tf.random_normal([2]), name='bias_o_cls'),
                           'bias_o_reg': tf.Variable(tf.random_normal([4]), name='bias_o_reg'),
                           'bias_o_fll': tf.Variable(tf.random_normal([10]), name='bias_o_fll')}
        conv_o_par_scale_par = {'bns_o_1': tf.Variable(tf.ones([32]), name='bns_o_1'),
                                'bns_o_2': tf.Variable(tf.ones([64]), name='bns_o_2'),
                                'bns_o_3': tf.Variable(tf.ones([64]), name='bns_o_3'),
                                'bns_o_4': tf.Variable(tf.ones([128]), name='bns_o_4'),
                                'bns_o_5': tf.Variable(tf.ones([256]), name='bns_o_5'),
                                'bns_o_cls': tf.Variable(tf.ones([2]), name='bns_o_cls'),
                                'bns_o_reg': tf.Variable(tf.ones([4]), name='bns_o_reg'),
                                'bns_o_fll': tf.Variable(tf.ones([10]), name='bns_o_fll')}
        conv_o_par_offset_par = {'bno_o_1': tf.Variable(tf.zeros([32]), name='bno_o_1'),
                                 'bno_o_2': tf.Variable(tf.zeros([64]), name='bno_o_2'),
                                 'bno_o_3': tf.Variable(tf.zeros([64]), name='bno_o_3'),
                                 'bno_o_4': tf.Variable(tf.zeros([128]), name='bno_o_4'),
                                 'bno_o_5': tf.Variable(tf.zeros([256]), name='bno_o_5'),
                                 'bno_o_cls': tf.Variable(tf.zeros([2]), name='bno_o_cls'),
                                 'bno_o_reg': tf.Variable(tf.zeros([4]), name='bno_o_reg'),
                                 'bno_o_fll': tf.Variable(tf.zeros([10]), name='bno_o_fll')}
        conv_o_par_me_EMA = {'conv_me_o_1': tf.Variable(tf.zeros([32]), trainable=False, name='conv_me_o_1'),
                             'conv_me_o_2': tf.Variable(tf.zeros([64]), trainable=False, name='conv_me_o_2'),
                             'conv_me_o_3': tf.Variable(tf.zeros([64]), trainable=False, name='conv_me_o_3'),
                             'conv_me_o_4': tf.Variable(tf.zeros([128]), trainable=False, name='conv_me_o_4'),
                             'conv_me_o_5': tf.Variable(tf.zeros([256]), trainable=False, name='conv_me_o_5'),
                             'conv_me_o_cls': tf.Variable(tf.zeros([2]), trainable=False, name='conv_me_o_cls'),
                             'conv_me_o_reg': tf.Variable(tf.zeros([4]), trainable=False, name='conv_me_o_reg'),
                             'conv_me_o_fll': tf.Variable(tf.zeros([10]), trainable=False, name='conv_me_o_fll')}
        conv_o_par_va_EMA = {'conv_va_o_1': tf.Variable(tf.ones([32]), trainable=False, name='conv_va_o_1'),
                             'conv_va_o_2': tf.Variable(tf.ones([64]), trainable=False, name='conv_va_o_2'),
                             'conv_va_o_3': tf.Variable(tf.ones([64]), trainable=False, name='conv_va_o_3'),
                             'conv_va_o_4': tf.Variable(tf.ones([128]), trainable=False, name='conv_va_o_4'),
                             'conv_va_o_5': tf.Variable(tf.ones([256]), trainable=False, name='conv_va_o_5'),
                             'conv_va_o_cls': tf.Variable(tf.ones([2]), trainable=False, name='conv_va_o_cls'),
                             'conv_va_o_reg': tf.Variable(tf.ones([4]), trainable=False, name='conv_va_o_reg'),
                             'conv_va_o_fll': tf.Variable(tf.ones([10]), trainable=False, name='conv_va_o_fll')}
    for par in [conv_o_par_weight, conv_o_par_bias, conv_o_par_scale_par, conv_o_par_offset_par, conv_o_par_me_EMA,
                conv_o_par_va_EMA]:
        for value in par.values():
            tf.add_to_collection('conv_o', value)


def normal(x, scale_p, offset_p, mean_p, var_p, on_train, decay, axes):
    # batch-normalization
    scale = scale_p
    offset = offset_p
    variance_epsilon = 0.0000001

    # moving average

    def mean_var_with_update():
        mean_ba, var_ba = tf.nn.moments(x, axes, name='moments')
        with tf.control_dependencies([assign_moving_average(mean_p, mean_ba, decay),
                                      assign_moving_average(var_p, var_ba, decay)]):
            return tf.identity(mean_ba), tf.identity(var_ba)

    # with tf.variable_scope('EMA'):
    mean, var = tf.cond(on_train, mean_var_with_update, lambda: (mean_p, var_p))

    return tf.nn.batch_normalization(x, mean, var, offset, scale, variance_epsilon)


def conv2d(x, W, b, scale_p, offset_p, mean_p, var_p, on_train, decay, conv_padding='SAME', stride=1):
    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=conv_padding)
    x = tf.nn.bias_add(x, b)
    # batch-normalization
    x = normal(x, scale_p, offset_p, mean_p, var_p, on_train, decay, axes=[0, 1, 2])
    return tf.nn.relu(x)


def maxpool2d(x, k_size=2, k_stride=2, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, k_size, k_size, 1], strides=[1, k_stride, k_stride, 1], padding=padding)


# conv-net creator
def conv_net(x, on_train, decay, out_put_fullconv_type):
    print('begin net')
    # x = tf.reshape(x, shape=[-1, 12, 12, 3])
    conv1 = conv2d(x, conv_p_par_weight['conv_p_1'], conv_p_par_bias['bias_p_1'], conv_p_par_scale_par['bns_p_1'],
                   conv_p_par_offset_par['bno_p_1'], conv_p_par_me_EMA['conv_me_p_1'], conv_p_par_va_EMA['conv_va_p_1'],
                   on_train,
                   decay, conv_padding='VALID')
    conv2 = maxpool2d(conv1, k_size=2, k_stride=2)
    conv2 = conv2d(conv2, conv_p_par_weight['conv_p_2'], conv_p_par_bias['bias_p_2'], conv_p_par_scale_par['bns_p_2'],
                   conv_p_par_offset_par['bno_p_2'], conv_p_par_me_EMA['conv_me_p_2'], conv_p_par_va_EMA['conv_va_p_2'],
                   on_train,
                   decay, conv_padding='VALID')
    conv3 = conv2d(conv2, conv_p_par_weight['conv_p_3'], conv_p_par_bias['bias_p_3'], conv_p_par_scale_par['bns_p_3'],
                   conv_p_par_offset_par['bno_p_3'], conv_p_par_me_EMA['conv_me_p_3'], conv_p_par_va_EMA['conv_va_p_3'],
                   on_train,
                   decay, conv_padding='VALID')
    # classify
    conv_cls = tf.nn.conv2d(conv3, conv_p_par_weight['conv_p_cls'], strides=[1, 1, 1, 1], padding='VALID')
    conv_cls = tf.nn.bias_add(conv_cls, conv_p_par_bias['bias_p_cls'])
    # batch-normalization
    conv_cls = normal(conv_cls, conv_p_par_scale_par['bns_p_cls'], conv_p_par_offset_par['bno_p_cls'],
                      conv_p_par_me_EMA['conv_me_p_cls'], conv_p_par_va_EMA['conv_va_p_cls'], on_train, decay,
                      axes=[0, 1, 2])
    conv_cls = tf.cond(out_put_fullconv_type, lambda: conv_cls, lambda: tf.reshape(conv_cls, [-1, 2]))
    out = tf.nn.softmax(conv_cls)
    # regression
    conv_reg = tf.nn.conv2d(conv3, conv_p_par_weight['conv_p_reg'], strides=[1, 1, 1, 1], padding='VALID')
    conv_reg = tf.nn.bias_add(conv_reg, conv_p_par_bias['bias_p_reg'])
    # batch-normalization
    conv_reg = normal(conv_reg, conv_p_par_scale_par['bns_p_reg'], conv_p_par_offset_par['bno_p_reg'],
                      conv_p_par_me_EMA['conv_me_p_reg'], conv_p_par_va_EMA['conv_va_p_reg'], on_train, decay,
                      axes=[0, 1, 2])
    conv_reg = tf.cond(out_put_fullconv_type, lambda: conv_reg, lambda: tf.reshape(conv_reg, [-1, 4]))

    return out, conv_reg


# conv-net creator
def conv_net_r(x, dropout, on_train, decay):
    print('begin net')
    conv1 = conv2d(x, conv_r_par_weight['conv_r_1'], conv_r_par_bias['bias_r_1'], conv_r_par_scale_par['bns_r_1'],
                   conv_r_par_offset_par['bno_r_1'], conv_r_par_me_EMA['conv_me_r_1'], conv_r_par_va_EMA['conv_va_r_1'],
                   on_train,
                   decay, conv_padding='VALID')
    conv2 = maxpool2d(conv1, k_size=3, k_stride=2, padding='SAME')
    conv2 = conv2d(conv2, conv_r_par_weight['conv_r_2'], conv_r_par_bias['bias_r_2'], conv_r_par_scale_par['bns_r_2'],
                   conv_r_par_offset_par['bno_r_2'], conv_r_par_me_EMA['conv_me_r_2'], conv_r_par_va_EMA['conv_va_r_2'],
                   on_train,
                   decay, conv_padding='VALID')
    conv3 = maxpool2d(conv2, k_size=3, k_stride=2, padding='VALID')
    conv3 = conv2d(conv3, conv_r_par_weight['conv_r_3'], conv_r_par_bias['bias_r_3'], conv_r_par_scale_par['bns_r_3'],
                   conv_r_par_offset_par['bno_r_3'], conv_r_par_me_EMA['conv_me_r_3'], conv_r_par_va_EMA['conv_va_r_3'],
                   on_train,
                   decay, conv_padding='VALID')
    fc4 = tf.reshape(conv3, [-1, conv_r_par_weight['full_r_4'].get_shape().as_list()[0]])
    fc4 = tf.matmul(fc4, conv_r_par_weight['full_r_4'])
    fc4 = tf.add(fc4, conv_r_par_bias['bias_r_4'])
    fc4 = normal(fc4, conv_r_par_scale_par['bns_r_4'], conv_r_par_offset_par['bno_r_4'],
                 conv_r_par_me_EMA['conv_me_r_4'], conv_r_par_va_EMA['conv_va_r_4'], on_train, decay,
                 axes=[0])
    fc4 = tf.nn.relu(fc4)
    fc4 = tf.nn.dropout(fc4, dropout)
    # classify
    fc_cls = tf.matmul(fc4, conv_r_par_weight['full_r_cls'])
    fc_cls = tf.add(fc_cls, conv_r_par_bias['bias_r_cls'])
    fc_cls = normal(fc_cls, conv_r_par_scale_par['bns_r_cls'], conv_r_par_offset_par['bno_r_cls'],
                    conv_r_par_me_EMA['conv_me_r_cls'], conv_r_par_va_EMA['conv_va_r_cls'], on_train, decay,
                    axes=[0])
    fc_cls = tf.nn.relu(fc_cls)
    fc_cls = tf.nn.softmax(fc_cls)
    # regression
    fc_reg = tf.matmul(fc4, conv_r_par_weight['full_r_reg'])
    fc_reg = tf.add(fc_reg, conv_r_par_bias['bias_r_reg'])
    fc_reg = normal(fc_reg, conv_r_par_scale_par['bns_r_reg'], conv_r_par_offset_par['bno_r_reg'],
                    conv_r_par_me_EMA['conv_me_r_reg'], conv_r_par_va_EMA['conv_va_r_reg'], on_train, decay,
                    axes=[0])

    return fc_cls, fc_reg


# conv-net creator
def conv_net_o(x, dropout, on_train, decay):
    print('begin net')
    conv1 = conv2d(x, conv_o_par_weight['conv_o_1'], conv_o_par_bias['bias_o_1'], conv_o_par_scale_par['bns_o_1'],
                   conv_o_par_offset_par['bno_o_1'], conv_o_par_me_EMA['conv_me_o_1'], conv_o_par_va_EMA['conv_va_o_1'],
                   on_train,
                   decay, conv_padding='VALID')
    conv2 = maxpool2d(conv1, k_size=3, k_stride=2, padding='SAME')
    conv2 = conv2d(conv2, conv_o_par_weight['conv_o_2'], conv_o_par_bias['bias_o_2'], conv_o_par_scale_par['bns_o_2'],
                   conv_o_par_offset_par['bno_o_2'], conv_o_par_me_EMA['conv_me_o_2'], conv_o_par_va_EMA['conv_va_o_2'],
                   on_train,
                   decay, conv_padding='VALID')
    conv3 = maxpool2d(conv2, k_size=3, k_stride=2, padding='VALID')
    conv3 = conv2d(conv3, conv_o_par_weight['conv_o_3'], conv_o_par_bias['bias_o_3'], conv_o_par_scale_par['bns_o_3'],
                   conv_o_par_offset_par['bno_o_3'], conv_o_par_me_EMA['conv_me_o_3'], conv_o_par_va_EMA['conv_va_o_3'],
                   on_train,
                   decay, conv_padding='VALID')
    conv4 = maxpool2d(conv3, k_size=2, k_stride=2, padding='VALID')
    conv4 = conv2d(conv4, conv_o_par_weight['conv_o_4'], conv_o_par_bias['bias_o_4'], conv_o_par_scale_par['bns_o_4'],
                   conv_o_par_offset_par['bno_o_4'], conv_o_par_me_EMA['conv_me_o_4'], conv_o_par_va_EMA['conv_va_o_4'],
                   on_train,
                   decay, conv_padding='VALID')
    fc5 = tf.reshape(conv4, [-1, conv_o_par_weight['full_o_5'].get_shape().as_list()[0]])
    fc5 = tf.matmul(fc5, conv_o_par_weight['full_o_5'])
    fc5 = tf.add(fc5, conv_o_par_bias['bias_o_5'])
    fc5 = normal(fc5, conv_o_par_scale_par['bns_o_5'], conv_o_par_offset_par['bno_o_5'],
                 conv_o_par_me_EMA['conv_me_o_5'], conv_o_par_va_EMA['conv_va_o_5'], on_train, decay,
                 axes=[0])
    fc5 = tf.nn.relu(fc5)
    fc5 = tf.nn.dropout(fc5, dropout)
    # classify
    fc_cls = tf.matmul(fc5, conv_o_par_weight['full_o_cls'])
    fc_cls = tf.add(fc_cls, conv_o_par_bias['bias_o_cls'])
    fc_cls = normal(fc_cls, conv_o_par_scale_par['bns_o_cls'], conv_o_par_offset_par['bno_o_cls'],
                    conv_o_par_me_EMA['conv_me_o_cls'], conv_o_par_va_EMA['conv_va_o_cls'], on_train, decay,
                    axes=[0])
    fc_cls = tf.nn.softmax(fc_cls)
    # regression
    fc_reg = tf.matmul(fc5, conv_o_par_weight['full_o_reg'])
    fc_reg = tf.add(fc_reg, conv_o_par_bias['bias_o_reg'])
    fc_reg = normal(fc_reg, conv_o_par_scale_par['bns_o_reg'], conv_o_par_offset_par['bno_o_reg'],
                    conv_o_par_me_EMA['conv_me_o_reg'], conv_o_par_va_EMA['conv_va_o_reg'], on_train, decay,
                    axes=[0])
    # facial landmark localization
    fc_fll = tf.matmul(fc5, conv_o_par_weight['full_o_fll'])
    fc_fll = tf.add(fc_fll, conv_o_par_bias['bias_o_fll'])
    fc_fll = normal(fc_fll, conv_o_par_scale_par['bns_o_fll'], conv_o_par_offset_par['bno_o_fll'],
                    conv_o_par_me_EMA['conv_me_o_fll'], conv_o_par_va_EMA['conv_va_o_fll'], on_train, decay,
                    axes=[0])
    return fc_cls, fc_reg, fc_fll


batch_size_pos = 26
batch_size_part = 26
batch_size_neg = 76
batch_size_fll = 52


# model training
def training_p_net():
    # training set
    tf.reset_default_graph()
    init_par_p()
    x = tf.placeholder(tf.float32, [None, 12, 12, 3])
    y = tf.placeholder(tf.float32, [None, 2])
    reg = tf.placeholder(tf.float32, [None, 4])
    indicator_det = tf.placeholder(tf.float32, [None, 1])
    indicator_reg = tf.placeholder(tf.float32, [None, 1])
    denote_det = tf.placeholder(tf.float32)
    denote_reg = tf.placeholder(tf.float32)
    out_put_fullconv_type = tf.placeholder(tf.bool)
    reg_train_data = np.zeros([batch_size_pos + batch_size_part + batch_size_neg, 4], dtype=np.float32)
    reg_val_data = np.zeros([batch_size_pos + batch_size_part + batch_size_neg, 4], dtype=np.float32)
    y_train_pos, x_train_pos, a_train_pos, GT_train_pos = read_TF_records.from_tfrecords_import_samples('train',
                                                                                                        batch_size_pos,
                                                                                                        None, 12, 'pos')
    y_train_part, x_train_part, a_train_part, GT_train_part = read_TF_records.from_tfrecords_import_samples('train',
                                                                                                            batch_size_part,
                                                                                                            None, 12,
                                                                                                            'part')
    y_train_neg, x_train_neg, a_train_neg, GT_train_neg = read_TF_records.from_tfrecords_import_samples('train',
                                                                                                        batch_size_neg,
                                                                                                        None, 12, 'neg')
    y_val_pos, x_val_pos, a_val_pos, GT_val_pos = read_TF_records.from_tfrecords_import_samples('val', batch_size_pos,
                                                                                                None, 12,
                                                                                                'pos')
    y_val_part, x_val_part, a_val_part, GT_val_part = read_TF_records.from_tfrecords_import_samples('val',
                                                                                                    batch_size_part,
                                                                                                    None, 12,
                                                                                                    'part')
    y_val_neg, x_val_neg, a_val_neg, GT_val_neg = read_TF_records.from_tfrecords_import_samples('val', batch_size_neg,
                                                                                                None, 12,
                                                                                                'neg')
    on_train = tf.placeholder(tf.bool)
    decay = tf.placeholder(tf.float32)
    pred, reg_a = conv_net(x, on_train, decay, out_put_fullconv_type)
    with tf.name_scope('loss'):
        # loss_batch = -tf.reduce_sum(y * tf.log(pred), reduction_indices=[1])
        # loss_batch = tf.reshape(loss_batch, [1, 128])
        # size_loss = tf.size(loss_batch)
        # sort_index = tf.nn.top_k(loss_batch, size_loss)[1]
        # ranking_loss = loss_batch[0, sort_index[0, 0]]
        # for i in range(10, 128):
        #     ranking_loss = tf.add(ranking_loss, loss_batch[0, sort_index[0, i]])
        # ranking_loss = tf.div(ranking_loss, top_num)
        test1 = tf.reduce_mean(
            -tf.reduce_sum(y * tf.log(pred), reduction_indices=[1], keepdims=True) * indicator_det * denote_det)
        test2 = tf.reduce_mean(tf.norm(reg - reg_a, axis=1, keepdims=True) * indicator_reg * denote_reg)
        loss = tf.reduce_mean(
            -tf.reduce_sum(y * tf.log(pred), reduction_indices=[1], keepdims=True) * indicator_det * denote_det +
            tf.norm(reg - reg_a, axis=1, keepdims=True) * indicator_reg * denote_reg)
        tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(0.005).minimize(loss)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('acc', accuracy)

    conv_vars = tf.get_collection('conv_p')
    saver_conv = tf.train.Saver(conv_vars)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs/logsold9_reg_like_rpn/', sess.graph)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        curr_ti = 0
        print('training...')
        indicator_det_data = np.ones([batch_size_pos + batch_size_part + batch_size_neg, 1])
        indicator_det_data[batch_size_pos:batch_size_pos + batch_size_part] = 0
        indicator_reg_data = np.ones([batch_size_pos + batch_size_part + batch_size_neg, 1])
        indicator_reg_data[
        batch_size_pos + batch_size_part:batch_size_pos + batch_size_part + batch_size_neg] = 0
        for i in range(120000):
            x_train_pos_data, y_train_pos_data, a_train_pos_data, GT_train_pos_data, x_train_part_data, y_train_part_data, a_train_part_data, GT_train_part_data, x_train_neg_data, y_train_neg_data, a_train_neg_data, GT_train_neg_data = sess.run(
                [x_train_pos, y_train_pos, a_train_pos, GT_train_pos, x_train_part, y_train_part, a_train_part,
                 GT_train_part, x_train_neg, y_train_neg, a_train_neg, GT_train_neg])
            x_train_data = np.append(x_train_pos_data, np.append(x_train_part_data, x_train_neg_data, axis=0),
                                     axis=0) / 255. - 0.5
            a_train_data = np.append(a_train_pos_data, np.append(a_train_part_data, a_train_neg_data, axis=0), axis=0)
            GT_train_data = np.append(GT_train_pos_data, np.append(GT_train_part_data, GT_train_neg_data, axis=0),
                                      axis=0)
            reg_train_data[:, 0] = (GT_train_data[:, 0] + GT_train_data[:, 2] / 2. - a_train_data[:, 0] -
                                    a_train_data[:, 2] / 2.) / a_train_data[:, 2].astype(np.float32)
            reg_train_data[:, 1] = (GT_train_data[:, 1] + GT_train_data[:, 3] / 2 - a_train_data[:, 1] -
                                    a_train_data[:, 3] / 2.) / a_train_data[:, 3].astype(np.float32)
            reg_train_data[:, 2] = np.log(GT_train_data[:, 2] / a_train_data[:, 2].astype(np.float32))
            reg_train_data[:, 3] = np.log(GT_train_data[:, 3] / a_train_data[:, 3].astype(np.float32))

            y_train_data = np.append(y_train_pos_data, np.append(y_train_part_data, y_train_neg_data, axis=0), axis=0)
            sess.run(optimizer,
                     feed_dict={x: x_train_data, y: y_train_data, reg: reg_train_data,
                                indicator_det: indicator_det_data, indicator_reg: indicator_reg_data, on_train: True,
                                decay: 0.5, denote_det: 1.25, denote_reg: 1.25,
                                out_put_fullconv_type: False})
            if i % 100 == 0:
                x_val_pos_data, y_val_pos_data, a_val_pos_data, GT_val_pos_data, x_val_part_data, y_val_part_data, a_val_part_data, GT_val_part_data, x_val_neg_data, y_val_neg_data, a_val_neg_data, GT_val_neg_data = sess.run(
                    [x_val_pos, y_val_pos, a_val_pos, GT_val_pos, x_val_part, y_val_part, a_val_part, GT_val_part,
                     x_val_neg, y_val_neg, a_val_neg, GT_val_neg])
                x_val_data = np.append(x_val_pos_data, np.append(x_val_part_data, x_val_neg_data, axis=0),
                                       axis=0) / 255. - 0.5
                a_val_data = np.append(a_val_pos_data, np.append(a_val_part_data, a_val_neg_data, axis=0),
                                       axis=0)
                GT_val_data = np.append(GT_val_pos_data, np.append(GT_val_part_data, GT_val_neg_data, axis=0),
                                        axis=0)
                reg_val_data[:, 0] = (GT_val_data[:, 0] + GT_val_data[:, 2] / 2. - a_val_data[:, 0] -
                                      a_val_data[:, 2] / 2.) / a_val_data[:, 2].astype(np.float32)
                reg_val_data[:, 1] = (GT_val_data[:, 1] + GT_val_data[:, 3] / 2 - a_val_data[:, 1] -
                                      a_val_data[:, 3] / 2.) / a_val_data[:, 3].astype(np.float32)
                reg_val_data[:, 2] = np.log(GT_val_data[:, 2] / a_val_data[:, 2].astype(np.float32))
                reg_val_data[:, 3] = np.log(GT_val_data[:, 3] / a_val_data[:, 3].astype(np.float32))
                y_val_data = np.append(y_val_pos_data, np.append(y_val_part_data, y_val_neg_data, axis=0),
                                       axis=0)
                t1, t2, rs, los, acc = sess.run([test1, test2, merged, loss, accuracy],
                                                feed_dict={x: x_val_data, y: y_val_data, reg: reg_val_data,
                                                           indicator_det: indicator_det_data,
                                                           indicator_reg: indicator_reg_data,
                                                           on_train: False,
                                                           decay: 0.5, denote_det: 1.25, denote_reg: 1.25,
                                                           out_put_fullconv_type: False})
                writer.add_summary(rs, i)
                print('the %d times,Loss=%f,acc=%f,spend:%fs' % (i, los, acc, time.clock() - curr_ti))
                curr_ti = time.clock()
        print('Save para in ', saver_conv.save(sess, './model/conv_p_net_12_100000.ckpt'))
        coord.request_stop()
        coord.join(threads)
    sess.close()


# model training
def training_r_net():
    # training set
    tf.reset_default_graph()
    init_par_r()
    x = tf.placeholder(tf.float32, [None, 24, 24, 3])
    y = tf.placeholder(tf.float32, [None, 2])
    reg = tf.placeholder(tf.float32, [None, 4])
    indicator_det = tf.placeholder(tf.float32, [None, 1])
    indicator_reg = tf.placeholder(tf.float32, [None, 1])
    denote_det = tf.placeholder(tf.float32)
    denote_reg = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    reg_train_data = np.zeros([batch_size_pos + batch_size_part + batch_size_neg, 4], dtype=np.float32)
    reg_val_data = np.zeros([batch_size_pos + batch_size_part + batch_size_neg, 4], dtype=np.float32)
    y_train_pos, x_train_pos, a_train_pos, GT_train_pos = read_TF_records.from_tfrecords_import_samples('train',
                                                                                                        batch_size_pos,
                                                                                                        None, 24, 'pos')
    y_train_part, x_train_part, a_train_part, GT_train_part = read_TF_records.from_tfrecords_import_samples('train',
                                                                                                            batch_size_part,
                                                                                                            None, 24,
                                                                                                            'part')
    y_train_neg, x_train_neg, a_train_neg, GT_train_neg = read_TF_records.from_tfrecords_import_samples('train',
                                                                                                        batch_size_neg,
                                                                                                        None, 24, 'neg')
    y_val_pos, x_val_pos, a_val_pos, GT_val_pos = read_TF_records.from_tfrecords_import_samples('val', batch_size_pos,
                                                                                                None, 24,
                                                                                                'pos')
    y_val_part, x_val_part, a_val_part, GT_val_part = read_TF_records.from_tfrecords_import_samples('val',
                                                                                                    batch_size_part,
                                                                                                    None, 24,
                                                                                                    'part')
    y_val_neg, x_val_neg, a_val_neg, GT_val_neg = read_TF_records.from_tfrecords_import_samples('val', batch_size_neg,
                                                                                                None, 24,
                                                                                                'neg')
    on_train = tf.placeholder(tf.bool)
    decay = tf.placeholder(tf.float32)
    fc_cls, fc_reg = conv_net_r(x, keep_prob, on_train, decay)
    with tf.name_scope('loss'):
        # loss_batch = -tf.reduce_sum(y * tf.log(pred), reduction_indices=[1])
        # loss_batch = tf.reshape(loss_batch, [1, 128])
        # size_loss = tf.size(loss_batch)
        # sort_index = tf.nn.top_k(loss_batch, size_loss)[1]
        # ranking_loss = loss_batch[0, sort_index[0, 0]]
        # for i in range(10, 128):
        #     ranking_loss = tf.add(ranking_loss, loss_batch[0, sort_index[0, i]])
        # ranking_loss = tf.div(ranking_loss, top_num)
        test1 = tf.reduce_mean(
            -tf.reduce_sum(y * tf.log(fc_cls), reduction_indices=[1], keepdims=True) * indicator_det * denote_det)
        test2 = tf.reduce_mean(tf.norm(reg - fc_reg, axis=1, keepdims=True) * indicator_reg * denote_reg)
        loss = tf.reduce_mean(
            -tf.reduce_sum(y * tf.log(fc_cls), reduction_indices=[1], keepdims=True) * indicator_det * denote_det +
            tf.norm(reg - fc_reg, axis=1, keepdims=True) * indicator_reg * denote_reg)
        tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(0.005).minimize(loss)

    correct_pred = tf.equal(tf.argmax(fc_cls, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('acc', accuracy)

    conv_vars = tf.get_collection('conv_r')
    saver_conv = tf.train.Saver(conv_vars)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs/logsoldr_reg_like_rpn/', sess.graph)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        curr_ti = 0
        print('training...')
        indicator_det_data = np.ones([batch_size_pos + batch_size_part + batch_size_neg, 1])
        indicator_det_data[batch_size_pos:batch_size_pos + batch_size_part] = 0
        indicator_reg_data = np.ones([batch_size_pos + batch_size_part + batch_size_neg, 1])
        indicator_reg_data[
        batch_size_pos + batch_size_part:batch_size_pos + batch_size_part + batch_size_neg] = 0
        for i in range(100000):
            x_train_pos_data, y_train_pos_data, a_train_pos_data, GT_train_pos_data, x_train_part_data, y_train_part_data, a_train_part_data, GT_train_part_data, x_train_neg_data, y_train_neg_data, a_train_neg_data, GT_train_neg_data = sess.run(
                [x_train_pos, y_train_pos, a_train_pos, GT_train_pos, x_train_part, y_train_part, a_train_part,
                 GT_train_part, x_train_neg, y_train_neg, a_train_neg, GT_train_neg])
            x_train_data = np.append(x_train_pos_data, np.append(x_train_part_data, x_train_neg_data, axis=0),
                                     axis=0) / 255. - 0.5
            a_train_data = np.append(a_train_pos_data, np.append(a_train_part_data, a_train_neg_data, axis=0), axis=0)
            GT_train_data = np.append(GT_train_pos_data, np.append(GT_train_part_data, GT_train_neg_data, axis=0),
                                      axis=0)
            reg_train_data[:, 0] = (GT_train_data[:, 0] + GT_train_data[:, 2] / 2. - a_train_data[:, 0] -
                                    a_train_data[:, 2] / 2.) / a_train_data[:, 2].astype(np.float32)
            reg_train_data[:, 1] = (GT_train_data[:, 1] + GT_train_data[:, 3] / 2 - a_train_data[:, 1] -
                                    a_train_data[:, 3] / 2.) / a_train_data[:, 3].astype(np.float32)
            reg_train_data[:, 2] = np.log(GT_train_data[:, 2] / a_train_data[:, 2].astype(np.float32))
            reg_train_data[:, 3] = np.log(GT_train_data[:, 3] / a_train_data[:, 3].astype(np.float32))

            y_train_data = np.append(y_train_pos_data, np.append(y_train_part_data, y_train_neg_data, axis=0), axis=0)
            sess.run(optimizer,
                     feed_dict={x: x_train_data, y: y_train_data, reg: reg_train_data,
                                indicator_det: indicator_det_data, indicator_reg: indicator_reg_data, on_train: True,
                                decay: 0.5, denote_det: 1.25, denote_reg: 1.25,
                                keep_prob: 0.5})
            if i % 100 == 0:
                x_val_pos_data, y_val_pos_data, a_val_pos_data, GT_val_pos_data, x_val_part_data, y_val_part_data, a_val_part_data, GT_val_part_data, x_val_neg_data, y_val_neg_data, a_val_neg_data, GT_val_neg_data = sess.run(
                    [x_val_pos, y_val_pos, a_val_pos, GT_val_pos, x_val_part, y_val_part, a_val_part, GT_val_part,
                     x_val_neg, y_val_neg, a_val_neg, GT_val_neg])
                x_val_data = np.append(x_val_pos_data, np.append(x_val_part_data, x_val_neg_data, axis=0),
                                       axis=0) / 255. - 0.5
                a_val_data = np.append(a_val_pos_data, np.append(a_val_part_data, a_val_neg_data, axis=0),
                                       axis=0)
                GT_val_data = np.append(GT_val_pos_data, np.append(GT_val_part_data, GT_val_neg_data, axis=0),
                                        axis=0)
                reg_val_data[:, 0] = (GT_val_data[:, 0] + GT_val_data[:, 2] / 2. - a_val_data[:, 0] -
                                      a_val_data[:, 2] / 2.) / a_val_data[:, 2].astype(np.float32)
                reg_val_data[:, 1] = (GT_val_data[:, 1] + GT_val_data[:, 3] / 2 - a_val_data[:, 1] -
                                      a_val_data[:, 3] / 2.) / a_val_data[:, 3].astype(np.float32)
                reg_val_data[:, 2] = np.log(GT_val_data[:, 2] / a_val_data[:, 2].astype(np.float32))
                reg_val_data[:, 3] = np.log(GT_val_data[:, 3] / a_val_data[:, 3].astype(np.float32))
                y_val_data = np.append(y_val_pos_data, np.append(y_val_part_data, y_val_neg_data, axis=0),
                                       axis=0)
                t1, t2, rs, los, acc = sess.run([test1, test2, merged, loss, accuracy],
                                                feed_dict={x: x_val_data, y: y_val_data, reg: reg_val_data,
                                                           indicator_det: indicator_det_data,
                                                           indicator_reg: indicator_reg_data,
                                                           on_train: False,
                                                           decay: 0.5, denote_det: 1.25, denote_reg: 1.25,
                                                           keep_prob: 1.})
                writer.add_summary(rs, i)
                print('the %d times,Loss=%f,acc=%f,spend:%fs' % (i, los, acc, time.clock() - curr_ti))
                curr_ti = time.clock()
        print('Save para in ', saver_conv.save(sess, './model/conv_r_net_12_100000.ckpt'))
        coord.request_stop()
        coord.join(threads)
    sess.close()


# model training
def training_o_net():
    # training set
    tf.reset_default_graph()
    init_par_o()
    x = tf.placeholder(tf.float32, [None, 48, 48, 3])
    y = tf.placeholder(tf.float32, [None, 2])
    reg = tf.placeholder(tf.float32, [None, 4])
    fll = tf.placeholder(tf.float32, [None, 10])
    indicator_det = tf.placeholder(tf.float32, [None, 1])
    indicator_reg = tf.placeholder(tf.float32, [None, 1])
    indicator_fll = tf.placeholder(tf.float32, [None, 1])
    denote_det = tf.placeholder(tf.float32)
    denote_reg = tf.placeholder(tf.float32)
    denote_fll = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    reg_train_data = np.zeros([batch_size_pos + batch_size_part + batch_size_neg + batch_size_fll, 4], dtype=np.float32)
    reg_val_data = np.zeros([batch_size_pos + batch_size_part + batch_size_neg + batch_size_fll, 4], dtype=np.float32)
    y_train_pos, x_train_pos, a_train_pos, GT_train_pos = read_TF_records.from_tfrecords_import_samples('train',
                                                                                                        batch_size_pos,
                                                                                                        None, 48, 'pos')
    y_train_part, x_train_part, a_train_part, GT_train_part = read_TF_records.from_tfrecords_import_samples('train',
                                                                                                            batch_size_part,
                                                                                                            None, 48,
                                                                                                            'part')
    y_train_neg, x_train_neg, a_train_neg, GT_train_neg = read_TF_records.from_tfrecords_import_samples('train',
                                                                                                        batch_size_neg,
                                                                                                        None, 48, 'neg')
    y_val_pos, x_val_pos, a_val_pos, GT_val_pos = read_TF_records.from_tfrecords_import_samples('val', batch_size_pos,
                                                                                                None, 48,
                                                                                                'pos')
    y_val_part, x_val_part, a_val_part, GT_val_part = read_TF_records.from_tfrecords_import_samples('val',
                                                                                                    batch_size_part,
                                                                                                    None, 48,
                                                                                                    'part')
    y_val_neg, x_val_neg, a_val_neg, GT_val_neg = read_TF_records.from_tfrecords_import_samples('val', batch_size_neg,
                                                                                                None, 48,
                                                                                                'neg')
    x_fll_train, landmark_train = read_TF_records.from_tfrecords_import_samples_lankmark('train', batch_size_fll, None)
    x_fll_val, landmark_val = read_TF_records.from_tfrecords_import_samples_lankmark('val', batch_size_fll, None)
    landmark_train = tf.reshape(landmark_train, [-1, 10])
    landmark_val = tf.reshape(landmark_val, [-1, 10])
    on_train = tf.placeholder(tf.bool)
    decay = tf.placeholder(tf.float32)
    fc_cls, fc_reg, fc_fll = conv_net_o(x, keep_prob, on_train, decay)
    with tf.name_scope('loss'):
        # loss_batch = -tf.reduce_sum(y * tf.log(pred), reduction_indices=[1])
        # loss_batch = tf.reshape(loss_batch, [1, 128])
        # size_loss = tf.size(loss_batch)
        # sort_index = tf.nn.top_k(loss_batch, size_loss)[1]
        # ranking_loss = loss_batch[0, sort_index[0, 0]]
        # for i in range(10, 128):
        #     ranking_loss = tf.add(ranking_loss, loss_batch[0, sort_index[0, i]])
        # ranking_loss = tf.div(ranking_loss, top_num)
        test1 = tf.reduce_mean(
            -tf.reduce_sum(y * tf.log(fc_cls), reduction_indices=[1], keepdims=True) * indicator_det * denote_det)
        test2 = tf.reduce_mean(tf.norm(reg - fc_reg, axis=1, keepdims=True) * indicator_reg * denote_reg)
        test3 = tf.reduce_mean(tf.norm(fll - fc_fll, axis=1, keepdims=True) * indicator_fll * denote_fll)
        loss = tf.reduce_mean(
            -tf.reduce_sum(y * tf.log(fc_cls), reduction_indices=[1], keepdims=True) * indicator_det * denote_det +
            tf.norm(reg - fc_reg, axis=1, keepdims=True) * indicator_reg * denote_reg +
            tf.norm(fll - fc_fll, axis=1, keepdims=True) * indicator_fll * denote_fll)
        tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(0.005).minimize(loss)
    acc_ob_fc_cls_pos = tf.slice(fc_cls, [0, 0], [batch_size_pos, 2])
    acc_ob_fc_cls_neg = tf.slice(fc_cls, [batch_size_pos + batch_size_part, 0],
                                 [batch_size_pos + batch_size_part + batch_size_neg, 2])
    acc_ob_y_pos = tf.slice(y, [0, 0], [batch_size_pos, 2])
    acc_ob_y_neg = tf.slice(y, [batch_size_pos + batch_size_part, 0],
                            [batch_size_pos + batch_size_part + batch_size_neg, 2])

    acc_ob_fc_cls = tf.concat([acc_ob_fc_cls_pos, acc_ob_fc_cls_neg], axis=0)
    acc_ob_y = tf.concat([acc_ob_y_pos, acc_ob_y_neg], axis=0)

    correct_pred = tf.equal(tf.argmax(acc_ob_fc_cls, 1), tf.argmax(acc_ob_y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('acc', accuracy)

    conv_vars = tf.get_collection('conv_o')
    saver_conv = tf.train.Saver(conv_vars)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs/logsoldo_landmark_like_rpn/', sess.graph)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        curr_ti = 0
        print('training...')
        indicator_det_data = np.ones([batch_size_pos + batch_size_part + batch_size_neg + batch_size_fll, 1])
        indicator_det_data[batch_size_pos:batch_size_pos + batch_size_part] = 0
        indicator_det_data[
        batch_size_pos + batch_size_part + batch_size_neg:
        batch_size_pos + batch_size_part + batch_size_neg + batch_size_fll] = 0

        indicator_reg_data = np.ones([batch_size_pos + batch_size_part + batch_size_neg + batch_size_fll, 1])
        indicator_reg_data[
        batch_size_pos + batch_size_part:batch_size_pos + batch_size_part + batch_size_neg + batch_size_fll] = 0

        indicator_fll_data = np.ones([batch_size_pos + batch_size_part + batch_size_neg + batch_size_fll, 1])
        indicator_fll_data[0:batch_size_pos + batch_size_part + batch_size_neg] = 0

        for i in range(50000):
            x_train_pos_data, y_train_pos_data, a_train_pos_data, GT_train_pos_data, x_train_part_data, y_train_part_data, a_train_part_data, GT_train_part_data, x_train_neg_data, y_train_neg_data, a_train_neg_data, GT_train_neg_data, x_fll_train_data, landmark_train_data = sess.run(
                [x_train_pos, y_train_pos, a_train_pos, GT_train_pos, x_train_part, y_train_part, a_train_part,
                 GT_train_part, x_train_neg, y_train_neg, a_train_neg, GT_train_neg, x_fll_train, landmark_train])
            x_train_data = np.append(np.append(x_train_pos_data, np.append(x_train_part_data, x_train_neg_data, axis=0),
                                               axis=0), x_fll_train_data, axis=0) / 255. - 0.5

            a_train_data = np.append(
                np.append(a_train_pos_data, np.append(a_train_part_data, a_train_neg_data, axis=0), axis=0),
                np.ones([batch_size_fll, 4], dtype=np.float32), axis=0)
            GT_train_data = np.append(
                np.append(GT_train_pos_data, np.append(GT_train_part_data, GT_train_neg_data, axis=0),
                          axis=0), np.ones([batch_size_fll, 4], dtype=np.float32), axis=0)
            reg_train_data[:, 0] = (GT_train_data[:, 0] + GT_train_data[:, 2] / 2. - a_train_data[:, 0] -
                                    a_train_data[:, 2] / 2.) / a_train_data[:, 2].astype(np.float32)
            reg_train_data[:, 1] = (GT_train_data[:, 1] + GT_train_data[:, 3] / 2 - a_train_data[:, 1] -
                                    a_train_data[:, 3] / 2.) / a_train_data[:, 3].astype(np.float32)
            reg_train_data[:, 2] = np.log(GT_train_data[:, 2] / a_train_data[:, 2].astype(np.float32))
            reg_train_data[:, 3] = np.log(GT_train_data[:, 3] / a_train_data[:, 3].astype(np.float32))
            reg_train_data[batch_size_pos + batch_size_part + batch_size_neg:] = 0.

            fll_train_data = np.append(
                np.zeros([batch_size_pos + batch_size_part + batch_size_neg, 10], dtype=np.float32),
                landmark_train_data, axis=0)

            y_train_data = np.append(
                np.append(y_train_pos_data, np.append(y_train_part_data, y_train_neg_data, axis=0), axis=0),
                np.zeros([batch_size_fll, 2], dtype=np.int8), axis=0)

            sess.run(optimizer,
                     feed_dict={x: x_train_data, y: y_train_data, reg: reg_train_data, fll: fll_train_data,
                                indicator_det: indicator_det_data, indicator_reg: indicator_reg_data,
                                indicator_fll: indicator_fll_data, on_train: True,
                                decay: 0.5, denote_det: 1.25, denote_reg: 1.25, denote_fll: 2.5 / 100.,
                                keep_prob: 0.5})
            # x_train_data[10:20] = x_train_data[150:160]
            tt_ffl, tt_fc_cls, tt_fc_reg, tt_1, tt_2, tt_3, tt_los = sess.run(
                [fc_fll, fc_cls, fc_reg, test1, test2, test3, loss],
                feed_dict={x: x_train_data, y: y_train_data,
                           reg: reg_train_data,
                           fll: fll_train_data,
                           indicator_det: indicator_det_data,
                           indicator_reg: indicator_reg_data,
                           indicator_fll: indicator_fll_data,
                           on_train: True,
                           decay: 0.5, denote_det: 1.25,
                           denote_reg: 1.25,
                           denote_fll: 2.5 / 100.,
                           keep_prob: 0.5})
            if i % 100 == 0:
                x_val_pos_data, y_val_pos_data, a_val_pos_data, GT_val_pos_data, x_val_part_data, y_val_part_data, a_val_part_data, GT_val_part_data, x_val_neg_data, y_val_neg_data, a_val_neg_data, GT_val_neg_data, x_fll_val_data, landmark_val_data = sess.run(
                    [x_val_pos, y_val_pos, a_val_pos, GT_val_pos, x_val_part, y_val_part, a_val_part, GT_val_part,
                     x_val_neg, y_val_neg, a_val_neg, GT_val_neg, x_fll_val, landmark_val])
                x_val_data = np.append(np.append(x_val_pos_data, np.append(x_val_part_data, x_val_neg_data, axis=0),
                                                 axis=0), x_fll_val_data, axis=0) / 255. - 0.5
                a_val_data = np.append(np.append(a_val_pos_data, np.append(a_val_part_data, a_val_neg_data, axis=0),
                                                 axis=0), np.ones([batch_size_fll, 4], dtype=np.float32), axis=0)
                GT_val_data = np.append(np.append(GT_val_pos_data, np.append(GT_val_part_data, GT_val_neg_data, axis=0),
                                                  axis=0), np.ones([batch_size_fll, 4], dtype=np.float32), axis=0)
                reg_val_data[:, 0] = (GT_val_data[:, 0] + GT_val_data[:, 2] / 2. - a_val_data[:, 0] -
                                      a_val_data[:, 2] / 2.) / a_val_data[:, 2].astype(np.float32)
                reg_val_data[:, 1] = (GT_val_data[:, 1] + GT_val_data[:, 3] / 2 - a_val_data[:, 1] -
                                      a_val_data[:, 3] / 2.) / a_val_data[:, 3].astype(np.float32)
                reg_val_data[:, 2] = np.log(GT_val_data[:, 2] / a_val_data[:, 2].astype(np.float32))
                reg_val_data[:, 3] = np.log(GT_val_data[:, 3] / a_val_data[:, 3].astype(np.float32))

                fll_val_data = np.append(
                    np.zeros([batch_size_pos + batch_size_part + batch_size_neg, 10], dtype=np.float32),
                    landmark_val_data, axis=0)
                y_val_data = np.append(np.append(y_val_pos_data, np.append(y_val_part_data, y_val_neg_data, axis=0),
                                                 axis=0), np.zeros([batch_size_fll, 2], dtype=np.int8), axis=0)
                rs, los, acc = sess.run([merged, loss, accuracy],
                                        feed_dict={x: x_val_data, y: y_val_data, reg: reg_val_data,
                                                   indicator_det: indicator_det_data,
                                                   indicator_reg: indicator_reg_data,
                                                   indicator_fll: indicator_fll_data,
                                                   fll: fll_val_data,
                                                   on_train: False,
                                                   decay: 0.5, denote_det: 1.25, denote_reg: 1.25,
                                                   denote_fll: 2.5 / 100.,
                                                   keep_prob: 1.})
                writer.add_summary(rs, i)
                print('the %d times,Loss=%f,acc=%f,spend:%fs' % (i, los, acc, time.clock() - curr_ti))
                curr_ti = time.clock()
        print('Save para in ', saver_conv.save(sess, './model/conv_o_net_12_landmark.ckpt'))
        coord.request_stop()
        coord.join(threads)
    sess.close()


# test for trained model
def from_ckpt_import_p_net():
    # training set
    tf.reset_default_graph()
    init_par_p()
    x = tf.placeholder(tf.float32, [None, 12, 12, 3])
    y = tf.placeholder(tf.float32, [None, 2])
    out_put_fullconv_type = tf.placeholder(tf.bool)
    y_val_pos, x_val_pos, _, _ = read_TF_records.from_tfrecords_import_samples('val', 1, None, 12, 'pos')
    y_val_neg, x_val_neg, _, _ = read_TF_records.from_tfrecords_import_samples('val', 127, None, 12, 'neg')
    on_train = tf.placeholder(tf.bool)
    decay = tf.placeholder(tf.float32)
    pred, _ = conv_net(x, on_train, decay, out_put_fullconv_type)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    conv_vars = tf.get_collection('conv_p')
    saver_conv = tf.train.Saver(conv_vars)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        saver_conv.restore(sess, './model/conv_p_net_12_100000.ckpt')
        curr_ti = 0
        print('testing...')
        entirety_acc = 0
        for i in range(100):
            x_val_pos_data, y_val_pos_data, x_val_neg_data, y_val_neg_data = sess.run(
                [x_val_pos, y_val_pos, x_val_neg, y_val_neg])
            x_val_data = np.append(x_val_pos_data, x_val_neg_data, axis=0) / 255. - 0.5
            y_val_data = np.append(y_val_pos_data, y_val_neg_data, axis=0)
            los, acc = sess.run([loss, accuracy],
                                feed_dict={x: x_val_data, y: y_val_data, on_train: False,
                                           decay: 0.5, out_put_fullconv_type: False})
            entirety_acc += acc
            print('the %d times,Loss=%f,acc=%f,spend:%fs' % (i, los, acc, time.clock() - curr_ti))
            curr_ti = time.clock()
        print('entirety acc is:%f' % entirety_acc)
        coord.request_stop()
        coord.join(threads)
    sess.close()


# test for trained model
def from_ckpt_import_r_net():
    # training set
    tf.reset_default_graph()
    init_par_r()
    x = tf.placeholder(tf.float32, [None, 24, 24, 3])
    y = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32)
    y_val_pos, x_val_pos, _, _ = read_TF_records.from_tfrecords_import_samples('val', 127, None, 24, 'pos')
    y_val_neg, x_val_neg, _, _ = read_TF_records.from_tfrecords_import_samples('val', 1, None, 24, 'neg')
    on_train = tf.placeholder(tf.bool)
    decay = tf.placeholder(tf.float32)
    pred, _ = conv_net_r(x, keep_prob, on_train, decay)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    conv_vars = tf.get_collection('conv_r')
    saver_conv = tf.train.Saver(conv_vars)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        saver_conv.restore(sess, './model/conv_r_net_12_100000.ckpt')
        curr_ti = 0
        print('testing...')
        entirety_acc = 0
        for i in range(100):
            x_val_pos_data, y_val_pos_data, x_val_neg_data, y_val_neg_data = sess.run(
                [x_val_pos, y_val_pos, x_val_neg, y_val_neg])
            x_val_data = np.append(x_val_pos_data, x_val_neg_data, axis=0) / 255. - 0.5
            y_val_data = np.append(y_val_pos_data, y_val_neg_data, axis=0)
            los, acc = sess.run([loss, accuracy],
                                feed_dict={x: x_val_data, y: y_val_data, on_train: False,
                                           decay: 0.5, keep_prob: 1.})
            entirety_acc += acc
            print('the %d times,Loss=%f,acc=%f,spend:%fs' % (i, los, acc, time.clock() - curr_ti))
            curr_ti = time.clock()
        print('entirety acc is:%f' % entirety_acc)
        coord.request_stop()
        coord.join(threads)
    sess.close()


# test for trained model
def from_ckpt_import_o_net():
    # training set
    tf.reset_default_graph()
    init_par_o()
    x = tf.placeholder(tf.float32, [None, 48, 48, 3])
    y = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32)
    images_landmark, _ = read_TF_records.from_tfrecords_import_samples_lankmark('train', 127, None)
    y_val_pos, x_val_pos, _, _ = read_TF_records.from_tfrecords_import_samples('val', 127, None, 48, 'pos')
    y_val_neg, x_val_neg, _, _ = read_TF_records.from_tfrecords_import_samples('val', 1, None, 48, 'neg')
    on_train = tf.placeholder(tf.bool)
    decay = tf.placeholder(tf.float32)
    pred, _, landmark = conv_net_o(x, keep_prob, on_train, decay)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    conv_vars = tf.get_collection('conv_o')
    saver_conv = tf.train.Saver(conv_vars)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        saver_conv.restore(sess, './model/conv_o_net_12_landmark.ckpt')
        curr_ti = 0
        print('testing...')
        entirety_acc = 0
        for i in range(100):
            x_val_pos_data, y_val_pos_data, x_val_neg_data, y_val_neg_data = sess.run(
                [images_landmark, y_val_pos, x_val_neg, y_val_neg])
            x_val_data = np.append(x_val_pos_data, x_val_neg_data, axis=0) / 255. - 0.5
            y_val_data = np.append(y_val_pos_data, y_val_neg_data, axis=0)
            los, acc, lanm = sess.run([loss, accuracy, landmark],
                                      feed_dict={x: x_val_data, y: y_val_data, on_train: False,
                                                 decay: 0.5, keep_prob: 1.})
            entirety_acc += acc
            print('the %d times,Loss=%f,acc=%f,spend:%fs' % (i, los, acc, time.clock() - curr_ti))
            curr_ti = time.clock()
        print('entirety acc is:%f' % entirety_acc)
        coord.request_stop()
        coord.join(threads)
    sess.close()


# test for landmark
def from_ckpt_testing_landmark_acc(im=[]):
    # training set
    tf.reset_default_graph()
    init_par_o()
    x = tf.placeholder(tf.float32, [None, 48, 48, 3])
    fll = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    _, x_val_pos, _, _ = read_TF_records.from_tfrecords_import_samples('val', 1, None, 48, 'pos')
    images_landmark, fll_land = read_TF_records.from_tfrecords_import_samples_lankmark('val', 1, None)
    fll_land = tf.reshape(fll_land, [-1, 10])
    on_train = tf.placeholder(tf.bool)
    decay = tf.placeholder(tf.float32)
    _, _, landmark = conv_net_o(x, keep_prob, on_train, decay)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.norm(fll - landmark, axis=1, keepdims=True))

    conv_vars = tf.get_collection('conv_o')
    saver_conv = tf.train.Saver(conv_vars)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        saver_conv.restore(sess, './model/conv_o_net_12_landmark.ckpt')
        curr_ti = 0
        print('testing...')
        entirety_los = 0
        for i in range(100):
            x_val_data1, fll_land_data1 = sess.run([images_landmark, fll_land])
            x_val_data2, fll_land_data2 = sess.run([x_val_pos, fll_land])
            # cv2.imwrite('./imgs/ob_1.jpg', x_val_data[0])
            if len(im) != 0:
                x_val_data1 = im
            x_val_data1 = x_val_data1 / 255. - 0.5
            los, land = sess.run([loss, landmark],
                                 feed_dict={x: x_val_data1, fll: fll_land_data1, on_train: False,
                                            decay: 0.5, keep_prob: 1.})
            im1 = x_val_data1[0] + 0.5
            for i in range(5):
                im1 = cv2.circle(im1, tuple(np.round(land[0, i * 2:i * 2 + 2]).astype(np.int32).tolist()), 1,
                                 (255., 0, 0), 2)
            cv2.imshow('test', im1)

            x_val_data2 = x_val_data2 / 255. - 0.5
            los, land = sess.run([loss, landmark],
                                 feed_dict={x: x_val_data2, fll: fll_land_data2, on_train: False,
                                            decay: 0.5, keep_prob: 1.})
            im2 = x_val_data2[0] + 0.5
            for i in range(5):
                im2 = cv2.circle(im2, tuple(np.round(land[0, i * 2:i * 2 + 2]).astype(np.int32).tolist()), 1,
                                 (255., 0, 0), 2)
            cv2.imshow('test2', im2)
            if cv2.waitKey() == ord('q'):
                break
            entirety_los += los
            print('the %d times,Loss=%f,spend:%fs' % (i, los, time.clock() - curr_ti))
            curr_ti = time.clock()
        print('entirety acc is:%f' % entirety_los)
        coord.request_stop()
        coord.join(threads)
    sess.close()


# from image cal it's feature map
# require:
# 1.image with 4 dimension
# return:
# 1.confidence feature map(3 dimension)
# 2.regression parameter(4 dimension)
def from_image_cal_p_net(im):
    # testing set
    tf.reset_default_graph()
    init_par_p()
    x = tf.placeholder(tf.float32, [None, im.shape[1], im.shape[2], 3])
    out_put_fullconv_type = tf.placeholder(tf.bool)
    on_train = tf.placeholder(tf.bool)
    decay = tf.placeholder(tf.float32)
    pred, reg_a = conv_net(x, on_train, decay, out_put_fullconv_type)

    conv_vars = tf.get_collection('conv_p')
    saver_conv = tf.train.Saver(conv_vars)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        saver_conv.restore(sess, './model/conv_p_net_12_100000.ckpt')
        print('testing...')
        feature_map, bbox_reg = sess.run([pred, reg_a],
                                         feed_dict={x: im, on_train: False, decay: 0.5, out_put_fullconv_type: True})
        conf_map = feature_map[:, :, :, 1]
        a = 1
        coord.request_stop()
        coord.join(threads)
    sess.close()
    return conf_map, bbox_reg


# get image pyramid
# require:
# 1.image(3 dimension)
# 2.min scale of pyramid
# 3.The scale of zoom
# return:
# image pyramid(3 dimension)
def from_im_import_pyramid_array(im, scale, zoom_scale):
    im_HEIGHT, im_WIDTH = im.shape[0:2]
    cur_h, cur_w = im_HEIGHT, im_WIDTH
    pyr_array = copy.deepcopy(im)
    cur_im = copy.deepcopy(im)
    pyr_array = pyr_array.reshape([1, im_HEIGHT, im_WIDTH, 3])
    while True:
        cur_h = int(cur_h / zoom_scale)
        cur_w = int(cur_w / zoom_scale)
        if min([cur_h, cur_w]) < scale:
            break
        template = np.zeros(im_HEIGHT * im_WIDTH * 3, dtype=np.uint8).reshape([1, im_HEIGHT, im_WIDTH, 3])
        cur_im = cv2.resize(cur_im, (cur_w, cur_h))
        template[0, 0:cur_h, 0:cur_w] = cur_im
        pyr_array = np.append(pyr_array, template, axis=0)
    return pyr_array


# get anchor rectangle
# require:
# 1.confidence feature map(3 dimension)
# 2.regression parameter(4 dimension)
# 3.The threshold of confident face
# 4.default scale of anchor rectangle
# 5.The scale of zoom
# 6.The threshold of iou
# 7.keep ROI square
# return:
# 1.The list of anchor rectangle with confident score:[[x,y,w,h,score,id]...]
def from_estimate_import_anchor_rect(conf_map, bbox_reg, threshold, scale_re, zoom_scale, iou_threshold, keep_square):
    anchor_rect = []
    for scale in range(conf_map.shape[0]):
        anchor_rect_scale = []
        regression_scale = []
        pos = np.where(conf_map[scale] > threshold)
        id = 0
        for pos_index in range(pos[0].shape[0]):
            x = pos[1][pos_index] * 2 * (zoom_scale ** scale)
            y = pos[0][pos_index] * 2 * (zoom_scale ** scale)
            w = scale_re * (zoom_scale ** scale)
            h = scale_re * (zoom_scale ** scale)
            reg_meta = bbox_reg[scale, pos[0][pos_index], pos[1][pos_index]].tolist()
            reg_meta.extend([id])
            regression_scale.append(reg_meta)
            anchor_rect_scale.append([x, y, w, h, conf_map[scale, pos[0][pos_index], pos[1][pos_index]], id])
            id += 1
        # local NMS
        anchor_rect_scale_NMS, _ = NMS_anchor_rect(anchor_rect_scale, iou_threshold)
        # regression
        for index in range(len(anchor_rect_scale_NMS)):
            for reg_meta in regression_scale:
                if anchor_rect_scale_NMS[index][5] == reg_meta[4]:
                    if keep_square:
                        reg_meta[2] = max(reg_meta[2:4])
                        reg_meta[3] = max(reg_meta[2:4])
                    anchor_rect_scale_NMS[index] = [
                        anchor_rect_scale_NMS[index][0] + anchor_rect_scale_NMS[index][2] * (
                                reg_meta[0] + (1 - np.exp(reg_meta[2])) / 2.),
                        anchor_rect_scale_NMS[index][1] + anchor_rect_scale_NMS[index][3] * (
                                reg_meta[1] + (1 - np.exp(reg_meta[3])) / 2.),
                        np.exp(reg_meta[2]) * anchor_rect_scale_NMS[index][2],
                        np.exp(reg_meta[3]) * anchor_rect_scale_NMS[index][3],
                        anchor_rect_scale_NMS[index][4], anchor_rect_scale_NMS[index][5]]
        anchor_rect.extend(anchor_rect_scale_NMS)
    return anchor_rect


# NMS operation
# require:
# 1.The list of anchor rectangle with confident score:[[x,y,w,h,score,id]...]
# 2.The threshold of iou
# 3.optional The landmark need to filter
# return:
# 1.The list of anchor rectangle with confident score after NMS:[[x,y,w,h,score,id]...]
# 2.Filtered landmark
def NMS_anchor_rect(anchor_rect, iou_threshold, landmark=[]):
    filtered_landmark = []
    anchor_rect_array = np.array(anchor_rect).reshape([-1, 6])
    score_array = anchor_rect_array[:, 4].reshape([-1, 1])
    filtered_anchor_rect = []
    while anchor_rect_array.shape[0] > 0:
        max_score_index = np.where(score_array == np.max(score_array, axis=0)[0])[0][0]
        check_arect = anchor_rect_array[max_score_index]
        same_one = False
        for far in filtered_anchor_rect:
            x = [far[0], far[0] + far[2], check_arect[0], check_arect[0] + check_arect[2]]
            y = [far[1], far[1] + far[3], check_arect[1], check_arect[1] + check_arect[3]]
            w_f = far[2]
            h_f = far[3]
            w_a = check_arect[2]
            h_a = check_arect[3]
            if x[1] <= x[2] or x[3] <= x[0] or y[3] <= y[0] or y[1] <= y[2]:
                intersection_area = 0.
            elif (x[2] - x[0]) * (x[3] - x[1]) < 0 and (y[2] - y[0]) * (y[3] - y[1]) < 0:
                same_one = True
            else:
                x.sort()
                y.sort()
                intersection_area = (x[2] - x[1]) * (y[2] - y[1])
                iou = float(intersection_area) / (w_f * h_f + w_a * h_a - intersection_area)
                if iou > iou_threshold:
                    same_one = True
        if not same_one:
            filtered_anchor_rect.append(check_arect.tolist())
            if not len(landmark) == 0:
                filtered_landmark.append(landmark[max_score_index])
        if not len(landmark) == 0:
            landmark = np.delete(landmark, max_score_index, axis=0)
        anchor_rect_array = np.delete(anchor_rect_array, max_score_index, axis=0)
        score_array = np.delete(score_array, max_score_index, axis=0)
    return filtered_anchor_rect, filtered_landmark


# show anchor rectangle in origin image
# require:
# 1.origin image(3 dimension)
# 2.The set of anchor rectangle with confident score:[[x,y,w,h,score,id]...]
# 3.The landmark of face
def from_anchor_rect_show_detective(im, anchor_rect, landmark):
    index = 0
    for rect in anchor_rect:
        im = cv2.rectangle(im, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])),
                           (0, 0, 255))
        if len(landmark) != 0:
            for i in range(5):
                im = cv2.circle(im, tuple((np.round(landmark[index][0, i * 2:i * 2 + 2]).astype(np.int32) + np.array(
                    [int(rect[0]), int(rect[1])]).reshape([1, 2])).tolist()[0]), 1, (255, 0, 0), 2)
            index += 1
    cv2.imshow('test', im)
    cv2.waitKey()


def filter_image_set_by_r_net(im, filter_name, anchor_rect, scale, filter_threshold, keep_square):
    if filter_name[0:6] == 'conv_r':
        print('r net')
        tf.reset_default_graph()
        init_par_r()
    elif filter_name[0:6] == 'conv_o':
        print('o net')
        tf.reset_default_graph()
        init_par_o()
    if len(anchor_rect) == 0:
        return anchor_rect
    scale_set = []
    im_HEIGHT, im_WIDTH = im.shape[0:2]
    zero_temp = np.zeros([3 * im_HEIGHT, 3 * im_WIDTH, 3], dtype=np.uint8)
    zero_temp[im_HEIGHT:2 * im_HEIGHT, im_WIDTH:2 * im_WIDTH] = im
    im = zero_temp
    im_rect = im[im_HEIGHT + int(anchor_rect[0][1]):im_HEIGHT + int(anchor_rect[0][1] +
                                                                    anchor_rect[0][3]),
              im_WIDTH + int(anchor_rect[0][0]):im_WIDTH + int(anchor_rect[0][0] +
                                                               anchor_rect[0][2])]

    scale_set.append(im_rect.shape[0] / float(scale))
    im_rect = cv2.resize(im_rect, (scale, scale))
    im_array = im_rect.reshape([1, scale, scale, 3])
    for rect in anchor_rect[1:]:
        rect_du = copy.deepcopy(rect)
        rect_du[0] = rect_du[0] + im_WIDTH
        rect_du[1] = rect_du[1] + im_HEIGHT
        im_rect = im[int(rect_du[1]):int(rect_du[1] + rect_du[3]), int(rect_du[0]):int(rect_du[0] + rect_du[2])]
        scale_set.append(im_rect.shape[0] / float(scale))
        im_rect = cv2.resize(im_rect, (scale, scale))
        im_array = np.append(im_array, im_rect.reshape([1, scale, scale, 3]), axis=0)
    im_array = im_array / 255. - 0.5

    # testing set
    x = tf.placeholder(tf.float32, [None, scale, scale, 3])
    on_train = tf.placeholder(tf.bool)
    decay = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    if filter_name[0:6] == 'conv_r':
        fc_cls, fc_reg = conv_net_r(x, keep_prob, on_train, decay)
    elif filter_name[0:6] == 'conv_o':
        fc_cls, fc_reg, fc_landmark = conv_net_o(x, keep_prob, on_train, decay)

    conv_vars = tf.get_collection(filter_name[0:6])
    saver_conv = tf.train.Saver(conv_vars)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        saver_conv.restore(sess, './model/%s.ckpt' % filter_name)
        print('testing...')
        if filter_name[0:6] == 'conv_r':
            cls, reg = sess.run([fc_cls, fc_reg],
                                feed_dict={x: im_array, on_train: False, decay: 0.5, keep_prob: 1.})
        elif filter_name[0:6] == 'conv_o':
            cls, reg, landmark = sess.run([fc_cls, fc_reg, fc_landmark],
                                          feed_dict={x: im_array, on_train: False, decay: 0.5, keep_prob: 1.})
        coord.request_stop()
        coord.join(threads)
    sess.close()
    # filter
    anchor_rect_filter = []
    landmark_filter = []
    for i in range(cls.shape[0]):
        if cls[i, 1] > filter_threshold:
            # regression
            if keep_square:
                reg[i, 2] = max(reg[i, 2:4])
                reg[i, 3] = max(reg[i, 2:4])
            x = anchor_rect[i][0] + anchor_rect[i][2] * (reg[i, 0] + (1 - np.exp(reg[i, 2])) / 2.)
            y = anchor_rect[i][1] + anchor_rect[i][3] * (reg[i, 1] + (1 - np.exp(reg[i, 3])) / 2.)
            w = np.exp(reg[i, 2]) * anchor_rect[i][2]
            h = np.exp(reg[i, 3]) * anchor_rect[i][3]
            anchor_rect_filter.append([x, y, w, h, anchor_rect[i][4], anchor_rect[i][5]])
            if filter_name[0:6] == 'conv_o':
                ori_landmark = landmark[i]
                offset_landmark = np.array([anchor_rect[i][2] * ((1 - np.exp(reg[i, 2])) / 2.),
                                            anchor_rect[i][3] * ((1 - np.exp(reg[i, 3])) / 2.),
                                            anchor_rect[i][2] * ((1 - np.exp(reg[i, 2])) / 2.),
                                            anchor_rect[i][3] * ((1 - np.exp(reg[i, 3])) / 2.),
                                            anchor_rect[i][2] * ((1 - np.exp(reg[i, 2])) / 2.),
                                            anchor_rect[i][3] * ((1 - np.exp(reg[i, 3])) / 2.),
                                            anchor_rect[i][2] * ((1 - np.exp(reg[i, 2])) / 2.),
                                            anchor_rect[i][3] * ((1 - np.exp(reg[i, 3])) / 2.),
                                            anchor_rect[i][2] * ((1 - np.exp(reg[i, 2])) / 2.),
                                            anchor_rect[i][3] * ((1 - np.exp(reg[i, 3])) / 2.),
                                            ]).reshape([1, 10])
                scale_landmark = np.array([np.exp(reg[i, 2]), np.exp(reg[i, 3]),
                                           np.exp(reg[i, 2]), np.exp(reg[i, 3]),
                                           np.exp(reg[i, 2]), np.exp(reg[i, 3]),
                                           np.exp(reg[i, 2]), np.exp(reg[i, 3]),
                                           np.exp(reg[i, 2]), np.exp(reg[i, 3])]).reshape([1, 10])
                landmark_filter.append((ori_landmark * scale_set[i]) * scale_landmark)

    return anchor_rect_filter, landmark_filter


def once_detective_pipline(im):
    filter_threshold_p = 0.98
    filter_threshold_r = 0.95
    filter_threshold_o = 0.95
    scale_P = 12
    scale_R = 24
    scale_o = 48
    zoom_scale_P = 1.2
    iou_threshold = 0.3
    im_pyr = from_im_import_pyramid_array(im, scale_P, zoom_scale_P) / 255. - 0.5
    conf_map, bbox_reg = from_image_cal_p_net(im_pyr)
    anchor_rect_p = from_estimate_import_anchor_rect(conf_map, bbox_reg, filter_threshold_p, scale_P, zoom_scale_P,
                                                     iou_threshold,
                                                     keep_square=True)
    anchor_rect_r, landmark_r = filter_image_set_by_r_net(im, 'conv_r_net_12_100000', anchor_rect_p, scale_R,
                                                          filter_threshold_r,
                                                          keep_square=True)
    anchor_rect_o, landmark_o = filter_image_set_by_r_net(im, 'conv_o_net_12_landmark', anchor_rect_r, scale_o,
                                                          filter_threshold_o,
                                                          keep_square=False)
    out, landmark_o_f = NMS_anchor_rect(anchor_rect_o, iou_threshold, landmark_o)
    from_anchor_rect_show_detective(im, out, landmark_o_f)


if __name__ == '__main__':
    # training_p_net()
    # from_ckpt_import_p_net()
    # training_r_net()
    # from_ckpt_import_r_net()
    # training_o_net()
    # from_ckpt_import_o_net()

    # im = cv2.imread('./imgs/test1.png')
    # test = np.zeros([48, 48, 3], dtype=np.uint8)
    # im = cv2.resize(im, (38, 38))
    # test[5:43, 5:43] = im
    # test=test.reshape([1, 48, 48, 3])
    # from_ckpt_testing_landmark_acc(test)

    im = cv2.imread('./imgs/img_18.jpg')
    once_detective_pipline(im)
    # test

    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print('error in open cap...')
    # while True:
    #     ret, im = cap.read()
    #     cv2.imshow('test', im)
    #     key = cv2.waitKey(30)
    #     if key == ord('q'):
    #         break
    #     elif key == ord('d'):
    #         once_detective_pipline(im)

    a = 1
