import tensorflow as tf


def conv2d_extraction(x, filters, size, strides=[1, 1], regularizer=None, activation=tf.nn.relu):
    return tf.layers.conv2d(inputs=x,
                            filters=filters,
                            strides=strides,
                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            padding='SAME',
                            kernel_size=size,
                            kernel_regularizer=regularizer)


def conv2d_reconstruction(x, filters, size, strides, regularizer=None, activation=tf.nn.relu):
    return tf.layers.conv2d_transpose(inputs=x,
                                      filters=filters,
                                      kernel_size=size,
                                      strides=strides,
                                      padding='SAME',
                                      kernel_regularizer=regularizer
                                        )


def dense(x, units):
    return tf.layers.dense(inputs=x,
                           units=units,
                           kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                           )


def encode_layer(x, count, size, stride, convolutions=1, regularizer=None, activation=tf.nn.relu, norm=True, maxpool=True):
    for i in range(0,convolutions):
        x = conv2d_extraction(x, count, size, [1, 1], regularizer=regularizer, activation=activation)
        if norm:
            x = tf.layers.batch_normalization(x,
                            fused=True)
        if activation:
            x = tf.nn.relu(x)
    if maxpool:
        x = pool(x, stride)
    return x


def decode_layer(x, count, size, stride, convolutions=0, regularizer=None, norm=True, activation=tf.nn.relu):
    deconv = conv2d_reconstruction(x, count, size, stride, activation=activation)
    for i in range(0,convolutions):
        deconv = conv2d_extraction(deconv, count, size, [1, 1], regularizer=regularizer)
        if norm:
            deconv = tf.layers.batch_normalization(deconv,
                            fused=True)
        if activation:
            deconv = tf.nn.relu(deconv)
    return deconv


def pool(x, strides=(2, 2)):
    return tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=strides,
        padding='same')

import tensorflow as tf

"""Takes a 256x256 image and reduces to 8x8xN, using shared weights"""

def siamese_encode(input, reuse=False):
    with tf.name_scope("siamese_encode"):
        with tf.variable_scope("siamese_1") as scope:
            filters_a = tf.layers.conv2d(input, 64, [3, 3], activation=tf.nn.relu, padding='SAME',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   reuse=reuse)
            filters_a = tf.layers.max_pooling2d(filters_a, pool_size=[2, 2], strides=[2, 2], padding='SAME')
        with tf.variable_scope("siamese_2") as scope:
            filters_b = tf.layers.conv2d(filters_a, 128, [3, 3], activation=tf.nn.relu, padding='SAME',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   reuse=reuse)
            filters_b = tf.layers.max_pooling2d(filters_b, pool_size=[2, 2], strides=[2, 2], padding='SAME')
        with tf.variable_scope("siamese_3") as scope:
            filters_c = tf.layers.conv2d(filters_b, 256, [3, 3], activation=tf.nn.relu, padding='SAME',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   reuse=reuse)
            filters_c = tf.layers.max_pooling2d(filters_c, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    return [filters_a, filters_b, filters_c]