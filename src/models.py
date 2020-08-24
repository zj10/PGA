from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import tflib as tl

conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = partial(tl.flatten_fully_connected, activation_fn=None)
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)


def conv_mnist():
    def Enc(img, z_dim, dim=64, use_bn=False, is_training=True, sigma=False):
        bn = partial(batch_norm, is_training=is_training) if use_bn else None
        conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)
        conv_lrelu = partial(conv, normalizer_fn=None, activation_fn=lrelu)

        with tf.variable_scope('Enc', reuse=tf.AUTO_REUSE):
            y = conv_lrelu(img, dim, 5, 2)
            y = conv_bn_lrelu(y, dim * 2, 5, 2)
            y = lrelu(fc(y, 1024, normalizer_fn=bn))
            z_mu = fc(y, z_dim)
            if sigma:
                z_log_sigma_sq = fc(y, z_dim, biases_initializer=tf.constant_initializer(2. * np.log(0.1)))
                return z_mu, z_log_sigma_sq
            else:
                return z_mu

    def Dec(z, dim=64, channels=1, use_bn=False, is_training=True):
        bn = partial(batch_norm, is_training=is_training) if use_bn else None
        dconv_relu = partial(dconv, normalizer_fn=None, activation_fn=relu)

        with tf.variable_scope('Dec', reuse=tf.AUTO_REUSE):
            y = relu(fc(z, 1024))
            y = relu(fc(y, 7 * 7 * dim * 2, normalizer_fn=bn))
            y = tf.reshape(y, [-1, 7, 7, dim * 2])
            y = dconv_relu(y, dim * 1, 5, 2)
            img = tf.tanh(dconv(y, channels, 5, 2))
            return img

    return Enc, Dec


def conv_32():
    def Enc(img, z_dim, dim=64, use_bn=False, is_training=True, sigma=False):
        bn = partial(batch_norm, is_training=is_training) if use_bn else None
        conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)
        conv_lrelu = partial(conv, normalizer_fn=None, activation_fn=lrelu)

        with tf.variable_scope('Enc', reuse=tf.AUTO_REUSE):
            y = conv_lrelu(img, dim, 5, 2)
            y = conv_bn_lrelu(y, dim * 2, 5, 2)
            y = conv_bn_lrelu(y, dim * 4, 5, 2)
            z_mu = fc(y, z_dim)
            if sigma:
                z_log_sigma_sq = fc(y, z_dim, biases_initializer=tf.constant_initializer(2. * np.log(0.1)))
                return z_mu, z_log_sigma_sq
            else:
                return z_mu

    def Dec(z, dim=64, channels=3, use_bn=False, is_training=True):
        bn = partial(batch_norm, is_training=is_training) if use_bn else None
        dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu)
        dconv_relu = partial(dconv, normalizer_fn=None, activation_fn=relu)

        with tf.variable_scope('Dec', reuse=tf.AUTO_REUSE):
            y = relu(fc(z, 4 * 4 * dim * 4))
            y = tf.reshape(y, [-1, 4, 4, dim * 4])
            y = dconv_bn_relu(y, dim * 2, 5, 2)
            y = dconv_relu(y, dim * 1, 5, 2)
            img = tf.tanh(dconv(y, channels, 5, 2))
            return img

    return Enc, Dec


def resnet_32():
    def res_block(inputs, out_channels, scale=None, use_bn=False, is_training=True, enc_first=False):
        assert scale in ['up', 'down', None]
        bn = partial(batch_norm, is_training=is_training) if use_bn else None
        conv1 = partial(conv, num_outputs=out_channels, kernel_size=1, stride=1)
        conv3 = partial(conv, num_outputs=out_channels, kernel_size=3, stride=1)

        skip, res = inputs, inputs
        if not enc_first:
            res = relu(bn(res)) if use_bn else relu(res)
        if scale == 'up':
            skip, res = tl.unpool(skip), tl.unpool(res)
        if out_channels != inputs.shape[-1] or scale:
            skip = conv1(skip)
        res = conv3(conv3(res, normalizer_fn=bn, activation_fn=relu))
        outputs = skip + res
        if scale == 'down':
            outputs = tl.pool(outputs)

        return outputs

    def Enc(img, z_dim, dim=512, use_bn=False, is_training=True, sigma=False):
        rb = partial(res_block, use_bn=use_bn, is_training=is_training)

        with tf.variable_scope('Enc', reuse=tf.AUTO_REUSE):
            y = rb(img, dim, 'down', enc_first=True)
            y = rb(y, dim, 'down')
            y = rb(y, dim)
            y = relu(rb(y, dim))
            z_mu = fc(y, z_dim)
            if sigma:
                z_log_sigma_sq = fc(y, z_dim, biases_initializer=tf.constant_initializer(2. * np.log(0.1)))
                return z_mu, z_log_sigma_sq
            else:
                return z_mu

    def Dec(z, dim=512, channels=3, use_bn=False, is_training=True):
        rb = partial(res_block, use_bn=use_bn, is_training=is_training)

        with tf.variable_scope('Dec', reuse=tf.AUTO_REUSE):
            y = fc(z, 4 * 4 * dim * 4)
            y = tf.reshape(y, [-1, 4, 4, dim * 4])
            y = rb(y, dim, 'up')
            y = rb(y, dim, 'up')
            y = rb(y, dim, 'up')
            img = tf.tanh(conv(relu(y), channels, 3, 1))
            return img

    return Enc, Dec


def conv_64():
    def Enc(img, z_dim, dim=64, use_bn=False, is_training=True, sigma=False):
        bn = partial(batch_norm, is_training=is_training) if use_bn else None
        conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)
        conv_lrelu = partial(conv, normalizer_fn=None, activation_fn=lrelu)

        with tf.variable_scope('Enc', reuse=tf.AUTO_REUSE):
            y = conv_lrelu(img, dim, 5, 2)
            y = conv_lrelu(y, dim * 2, 5, 2)
            y = conv_bn_lrelu(y, dim * 4, 5, 2)
            y = conv_bn_lrelu(y, dim * 8, 5, 2)
            z_mu = fc(y, z_dim)
            if sigma:
                z_log_sigma_sq = fc(y, z_dim, biases_initializer=tf.constant_initializer(2. * np.log(0.1)))
                return z_mu, z_log_sigma_sq
            else:
                return z_mu

    def Dec(z, dim=64, channels=3, use_bn=False, is_training=True):
        bn = partial(batch_norm, is_training=is_training) if use_bn else None
        dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu)
        dconv_relu = partial(dconv, normalizer_fn=None, activation_fn=relu)

        with tf.variable_scope('Dec', reuse=tf.AUTO_REUSE):
            y = relu(fc(z, 4 * 4 * dim * 8))
            y = tf.reshape(y, [-1, 4, 4, dim * 8])
            y = dconv_bn_relu(y, dim * 4, 5, 2)
            y = dconv_bn_relu(y, dim * 2, 5, 2)
            y = dconv_relu(y, dim * 1, 5, 2)
            img = tf.tanh(dconv(y, channels, 5, 2))
            return img

    return Enc, Dec
