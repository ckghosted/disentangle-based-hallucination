import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)

class batch_norm_old(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name
    def __call__(self, x, train=True, reuse=False):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name,
                                            reuse=reuse)
# def batch_norm(x, is_train, name='bn'):
#     with tf.variable_scope(name):
#         return tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5, training=is_train)
# https://github.com/dalgu90/resnet-18-tensorflow
# https://stackoverflow.com/questions/48001759/what-is-right-batch-normalization-function-in-tensorflow/48006315#48006315
def batch_norm(x, is_train, global_step=None, name='bn'):
    moving_average_decay = 0.9
    # moving_average_decay = 0.99
    # moving_average_decay_init = 0.99
    with tf.variable_scope(name):
        decay = moving_average_decay
        # if global_step is None:
            # decay = moving_average_decay
        # else:
            # decay = tf.cond(tf.greater(global_step, 100)
                            # , lambda: tf.constant(moving_average_decay, tf.float32)
                            # , lambda: tf.constant(moving_average_decay_init, tf.float32))
        input_shape = x.get_shape().as_list()
        if len(input_shape) == 4:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        elif len(input_shape) == 2:
            batch_mean, batch_var = tf.nn.moments(x, [0])
        with tf.device('/CPU:0'):
            mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer(), trainable=False)
            sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer(), trainable=False)
            beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer())
        # BN when training
        update = 1.0 - decay
        # with tf.control_dependencies([tf.Print(decay, [decay])]):
            # update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_sigma = sigma.assign_sub(update*(sigma - batch_var))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

        mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                            lambda: (mu, sigma))
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

        # bn = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-5)

        # bn = tf.contrib.layers.batch_norm(inputs=x, decay=decay,
                                          # updates_collections=[tf.GraphKeys.UPDATE_OPS], center=True,
                                          # scale=True, epsilon=1e-5, is_training=is_train,
                                          # trainable=True)
        return bn

def instance_norm(x, name='instance_norm'):
    return tf.contrib.layers.instance_norm(x,
                                           epsilon=1e-5,
                                           center=True, scale=True,
                                           scope=name)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           add_bias=False,
           padding='SAME',
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        if add_bias:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            conv = tf.nn.bias_add(conv, biases)
        
        return conv

def deconv2d(input_, output_dim,
             k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
             add_bias=False,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_dim[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_dim, strides=[1, d_h, d_w, 1])
        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_dim, strides=[1, d_h, d_w, 1])
        
        if add_bias:
            biases = tf.get_variable('biases', [output_dim[-1]], initializer=tf.constant_initializer(0.0))
            # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
            deconv = tf.nn.bias_add(deconv, biases)
        
        if with_w and add_bias:
            return deconv, w, biases
        elif with_w:
            return deconv, w
        else:
            return deconv

def max_pool(x, name, ksize=3, strides=2):
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding='SAME', name=name)

def prelu(x, name="prelu"):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alphas * (x - abs(x)) * 0.5
        return pos + neg

def lrelu(x, leak=0.01, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, stddev=0.02, bias_start=0.0,
           add_bias=False,
           name="linear", with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        if len(shape) == 2:
            matrix = tf.get_variable("Matrix", [shape[-1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
            if add_bias:
                bias = tf.get_variable("bias", [output_size],
                                       initializer=tf.constant_initializer(bias_start))
            if add_bias:
                if with_w:
                    return tf.matmul(input_, matrix) + bias, matrix, bias
                else:
                    return tf.matmul(input_, matrix) + bias
            else:
                if with_w:
                    return tf.matmul(input_, matrix), matrix
                else:
                    return tf.matmul(input_, matrix)
        elif len(shape) == 3:
            matrix = tf.get_variable("Matrix", [output_size, shape[-1], 1], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
            if add_bias:
                bias = tf.get_variable("bias", [output_size, 1],
                                       initializer=tf.constant_initializer(bias_start))
            if add_bias:
                if with_w:
                    return tf.matmul(input_, matrix) + tf.reshape(tf.tile(bias, [shape[1], 1]), [-1, shape[1], 1]), matrix, bias
                else:
                    return tf.matmul(input_, matrix) + tf.reshape(tf.tile(bias, [shape[1], 1]), [-1, shape[1], 1])
            else:
                if with_w:
                    return tf.matmul(input_, matrix), matrix
                else:
                    return tf.matmul(input_, matrix)

## linear layer with the matrix being initialized by identity matrix (maybe some zero rows or columns if not square matrix)
def linear_identity(input_, output_size, stddev=0.02, bias_start=0.0,
           add_bias=False,
           name="linear", with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        if len(shape) == 2:
            if shape[-1] >= output_size:
                init_matrix = np.concatenate((np.identity(output_size), np.zeros((shape[-1] - output_size, output_size))), axis=0)
            else:
                init_matrix = np.identity(output_size)[0:shape[-1], :]
            matrix = tf.get_variable("Matrix", [shape[-1], output_size], tf.float32,
                                     initializer=tf.constant_initializer(init_matrix))
            if add_bias:
                bias = tf.get_variable("bias", [output_size],
                                       initializer=tf.constant_initializer(bias_start))
            if add_bias:
                if with_w:
                    return tf.matmul(input_, matrix) + bias, matrix, bias
                else:
                    return tf.matmul(input_, matrix) + bias
            else:
                if with_w:
                    return tf.matmul(input_, matrix), matrix
                else:
                    return tf.matmul(input_, matrix)
        elif len(shape) == 3:
            matrix = tf.get_variable("Matrix", [output_size, shape[-1], 1], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
            if add_bias:
                bias = tf.get_variable("bias", [output_size, 1],
                                       initializer=tf.constant_initializer(bias_start))
            if add_bias:
                if with_w:
                    return tf.matmul(input_, matrix) + tf.reshape(tf.tile(bias, [shape[1], 1]), [-1, shape[1], 1]), matrix, bias
                else:
                    return tf.matmul(input_, matrix) + tf.reshape(tf.tile(bias, [shape[1], 1]), [-1, shape[1], 1])
            else:
                if with_w:
                    return tf.matmul(input_, matrix), matrix
                else:
                    return tf.matmul(input_, matrix)

# https://github.com/dalgu90/resnet-18-tensorflow
# Two kinds of 'SimpleBlock' in CloserLookFewShot/backbone.py
def resblk(x, is_train, with_BN=True, name='resblk'):
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        # Shortcut connection
        shortcut = x
        # Residual
        x = conv2d(x, output_dim=num_channel, k_h=3, k_w=3, d_h=1, d_w=1, add_bias=(~with_BN), name='conv_1')
        if with_BN:
            x = batch_norm(x, is_train=is_train, name='bn_1')
        x = tf.nn.relu(x, name='relu_1')
        x = conv2d(x, output_dim=num_channel, k_h=3, k_w=3, d_h=1, d_w=1, add_bias=(~with_BN), name='conv_2')
        if with_BN:
            x = batch_norm(x, is_train=is_train, name='bn_2')
        # Merge
        x = x + shortcut
        x = tf.nn.relu(x, name='relu_2')
    return x
def resblk_IN(x, name='resblk_IN'):
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        # Shortcut connection
        shortcut = x
        # Residual
        x = conv2d(x, output_dim=num_channel, k_h=3, k_w=3, d_h=1, d_w=1, name='conv_1')
        x = instance_norm(x, name='ins_norm_1')
        x = tf.nn.relu(x, name='relu_1')
        x = conv2d(x, output_dim=num_channel, k_h=3, k_w=3, d_h=1, d_w=1, name='conv_2')
        x = instance_norm(x, name='ins_norm_2')
        # Merge
        x = x + shortcut
        x = tf.nn.relu(x, name='relu_2')
    return x
def resblk_first(x, out_channel, kernels, strides, is_train, with_BN=True, name='resblk_first'):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, ksize=[1, strides, strides, 1], strides=[1, strides, strides, 1], padding='VALID')
            else:
                shortcut = conv2d(x, output_dim=out_channel, k_h=1, k_w=1, d_h=strides, d_w=strides, add_bias=(~with_BN), name='shortcut')
                if with_BN:
                    shortcut = batch_norm(shortcut, is_train=is_train, name='bn_shortcut')
            # Residual
            x = conv2d(x, output_dim=out_channel, k_h=kernels, k_w=kernels, d_h=strides, d_w=strides, add_bias=(~with_BN), name='conv_1')
            if with_BN:
                x = batch_norm(x, is_train=is_train, name='bn_1')
            x = tf.nn.relu(x, name='relu_1')
            x = conv2d(x, output_dim=out_channel, k_h=kernels, k_w=kernels, d_h=1, d_w=1, add_bias=(~with_BN), name='conv_2')
            if with_BN:
                x = batch_norm(x, is_train=is_train, name='bn_2')
            # Merge
            x = x + shortcut
            x = tf.nn.relu(x, name='relu_2')
        return x
def resblk_first_IN(x, out_channel, kernels, strides, name='resblk_first_IN'):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, ksize=[1, strides, strides, 1], strides=[1, strides, strides, 1], padding='VALID')
            else:
                shortcut = conv2d(x, output_dim=out_channel, k_h=1, k_w=1, d_h=strides, d_w=strides, name='shortcut')
                shortcut = instance_norm(shortcut, name='ins_noem_shortcut')
            # Residual
            x = conv2d(x, output_dim=out_channel, k_h=kernels, k_w=kernels, d_h=strides, d_w=strides, name='conv_1')
            x = instance_norm(x, name='ins_norm_1')
            x = tf.nn.relu(x, name='relu_1')
            x = conv2d(x, output_dim=out_channel, k_h=kernels, k_w=kernels, d_h=1, d_w=1, name='conv_2')
            x = instance_norm(x, name='ins_norm_2')
            # Merge
            x = x + shortcut
            x = tf.nn.relu(x, name='relu_2')
        return x

# https://github.com/piyush2896/ResNet50-Tensorflow
# Two kinds of 'BottleneckBlock' in CloserLookFewShot/backbone.py
def idenblk(x, out_channels, is_train, with_BN=True, name='idenblk'):
        with tf.variable_scope(name) as scope:
            # Shortcut connection
            shortcut = x
            # Residual
            x = conv2d(x, output_dim=out_channels[0], k_h=1, k_w=1, d_h=1, d_w=1, add_bias=(~with_BN), padding='VALID', name='conv_1')
            if with_BN:
                x = batch_norm(x, is_train=is_train, name='bn_1')
            x = tf.nn.relu(x, name='relu_1')
            x = conv2d(x, output_dim=out_channels[1], k_h=3, k_w=3, d_h=1, d_w=1, add_bias=(~with_BN), padding='SAME', name='conv_2')
            if with_BN:
                x = batch_norm(x, is_train=is_train, name='bn_2')
            x = tf.nn.relu(x, name='relu_2')
            x = conv2d(x, output_dim=out_channels[2], k_h=1, k_w=1, d_h=1, d_w=1, add_bias=(~with_BN), padding='VALID', name='conv_3')
            if with_BN:
                x = batch_norm(x, is_train=is_train, name='bn_3')
            # Merge
            x = x + shortcut
            x = tf.nn.relu(x, name='relu_merge')
        return x
def idenblk_IN(x, out_channels, name='idenblk_IN'):
        with tf.variable_scope(name) as scope:
            # Shortcut connection
            shortcut = x
            # Residual
            x = conv2d(x, output_dim=out_channels[0], k_h=1, k_w=1, d_h=1, d_w=1, padding='VALID', name='conv_1')
            x = instance_norm(x, name='ins_norm_1')
            x = tf.nn.relu(x, name='relu_1')
            x = conv2d(x, output_dim=out_channels[1], k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name='conv_2')
            x = instance_norm(x, name='ins_norm_2')
            x = tf.nn.relu(x, name='relu_2')
            x = conv2d(x, output_dim=out_channels[2], k_h=1, k_w=1, d_h=1, d_w=1, padding='VALID', name='conv_3')
            x = instance_norm(x, name='ins_norm_3')
            # Merge
            x = x + shortcut
            x = tf.nn.relu(x, name='relu_merge')
        return x
# Having stride=2 in the middle layer (instead of the first layer) based on 'BottleneckBlock' in CloserLookFewShot/backbone.py
def convblk(x, out_channels, stride_first, stride_middle, is_train, with_BN=True, name='convblk'):
        with tf.variable_scope(name) as scope:
            # Shortcut connection
            # shortcut = conv2d(x, output_dim=out_channels[2], k_h=1, k_w=1, d_h=stride_first, d_w=stride_first, padding='VALID', name='conv_shortcut')
            shortcut = conv2d(x, output_dim=out_channels[2], k_h=1, k_w=1, d_h=stride_middle, d_w=stride_middle, add_bias=(~with_BN), padding='VALID', name='conv_shortcut')
            if with_BN:
                shortcut = batch_norm(shortcut, is_train=is_train, name='bn_shortcut')
            # Residual
            x = conv2d(x, output_dim=out_channels[0], k_h=1, k_w=1, d_h=stride_first, d_w=stride_first, add_bias=(~with_BN), padding='VALID', name='conv_1')
            if with_BN:
                x = batch_norm(x, is_train=is_train, name='bn_1')
            x = tf.nn.relu(x, name='relu_1')
            x = conv2d(x, output_dim=out_channels[1], k_h=3, k_w=3, d_h=stride_middle, d_w=stride_middle, add_bias=(~with_BN), padding='SAME', name='conv_2')
            if with_BN:
                x = batch_norm(x, is_train=is_train, name='bn_2')
            x = tf.nn.relu(x, name='relu_2')
            x = conv2d(x, output_dim=out_channels[2], k_h=1, k_w=1, d_h=1, d_w=1, add_bias=(~with_BN), padding='VALID', name='conv_3')
            if with_BN:
                x = batch_norm(x, is_train=is_train, name='bn_3')
            # Merge
            x = x + shortcut
            x = tf.nn.relu(x, name='relu_merge')
        return x
def convblk_IN(x, out_channels, stride_first, stride_middle, name='convblk_IN'):
        with tf.variable_scope(name) as scope:
            # Shortcut connection
            # shortcut = conv2d(x, output_dim=out_channels[2], k_h=1, k_w=1, d_h=stride_first, d_w=stride_first, padding='VALID', name='conv_shortcut')
            shortcut = conv2d(x, output_dim=out_channels[2], k_h=1, k_w=1, d_h=stride_middle, d_w=stride_middle, padding='VALID', name='conv_shortcut')
            shortcut = instance_norm(shortcut, name='ins_norm_shortcut')
            # Residual
            x = conv2d(x, output_dim=out_channels[0], k_h=1, k_w=1, d_h=stride_first, d_w=stride_first, padding='VALID', name='conv_1')
            x = instance_norm(x, name='ins_norm_1')
            x = tf.nn.relu(x, name='relu_1')
            x = conv2d(x, output_dim=out_channels[1], k_h=3, k_w=3, d_h=stride_middle, d_w=stride_middle, padding='SAME', name='conv_2')
            x = instance_norm(x, name='ins_norm_2')
            x = tf.nn.relu(x, name='relu_2')
            x = conv2d(x, output_dim=out_channels[2], k_h=1, k_w=1, d_h=1, d_w=1, padding='VALID', name='conv_3')
            x = instance_norm(x, name='ins_norm_3')
            # Merge
            x = x + shortcut
            x = tf.nn.relu(x, name='relu_merge')
        return x