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


def encode_layer(x, count, size, stride, convolutions=1, regularizer=None, activation=tf.nn.relu, norm=True,
                 maxpool=True):
    for i in range(0, convolutions):
        x = conv2d_extraction(x, count, size, [1, 1], regularizer=regularizer, activation=activation)
        if norm:
            x = tf.layers.batch_normalization(x,
                                              fused=False)
        if activation:
            x = tf.nn.relu(x)
    if maxpool:
        x = pool(x, stride)
    return x


def encode_layer_siamese(x, convolutions, filters, kernel_size, reuse, name):
    for i in range(0, convolutions):
        with tf.variable_scope("{}_{}".format(name,i)) as scope:
            x = tf.layers.conv2d(x, filters, kernel_size, activation=None, padding='SAME',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 reuse=reuse)
            x = tf.layers.batch_normalization(x,
                                              fused=False, reuse=reuse)
            x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    return x


def decode_layer(x, count, size, stride, convolutions=0, regularizer=None, norm=True, activation=tf.nn.relu):
    deconv = conv2d_reconstruction(x, count, size, stride, activation=activation)
    for i in range(0, convolutions):
        deconv = conv2d_extraction(deconv, count, size, [1, 1], regularizer=regularizer)
        if norm:
            deconv = tf.layers.batch_normalization(deconv,
                                                   fused=False)
        if activation:
            deconv = tf.nn.relu(deconv)
    return deconv


def pool(x, strides=(2, 2)):
    return tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=strides,
        padding='same')

"""Takes a 256x256 image and reduces to 8x8xN, using shared weights"""


def siamese_encode(input, reuse=False):
    with tf.name_scope("siamese_encode"):
        filters_a = encode_layer_siamese(input, 3, 64, [3,3], reuse, "siamese_1")
        filters_b = encode_layer_siamese(filters_a, 3, 128, [3,3], reuse, "siamese_2")
        filters_c = encode_layer_siamese(filters_b, 3, 256, [3,3], reuse, "siamese_3")
    return [filters_a, filters_b, filters_c]

def siamese_encode_2(input, reuse=False, depth=3):
    with tf.name_scope("siamese_encode_v2"):
        filters_a = encode_layer_siamese(input, depth, 64, [3,3], reuse, "siamese_1")
        filters_b = encode_layer_siamese(filters_a, depth, 128, [3,3], reuse, "siamese_2")
        filters_c = encode_layer_siamese(filters_b, depth, 256, [3,3], reuse, "siamese_3")
        downsampled_a = tf.layers.max_pooling2d(filters_a, pool_size=[4, 4], strides=[4, 4], padding='SAME')
        downsampled_b = tf.layers.max_pooling2d(filters_b, pool_size=[2, 2], strides=[2, 2], padding='SAME')
        multiscale_c = tf.layers.max_pooling2d(filters_c, pool_size=[4, 4], strides=[4, 4], padding='SAME')
        multiscale_b = tf.layers.max_pooling2d(downsampled_b, pool_size=[2, 2], strides=[2, 2], padding='SAME')
        multiscale_a = downsampled_a

    return  filters_c, [downsampled_a, downsampled_b], [multiscale_a, multiscale_b, multiscale_c]

def basic_encode(input, depth=3, name=""):
    filters_a = encode_layer_siamese(input, depth, 64, [3,3],False, "basic_1_{}".format(name))
    filters_b = encode_layer_siamese(filters_a, depth, 128, [3,3],False, "basic_2_{}".format(name))
    filters_c = encode_layer_siamese(filters_b, depth , 256, [3,3],False, "basic_3_{}".format(name))
    filters_d = encode_layer_siamese(filters_c, depth , 512, [3,3],False, "basic_4_{}".format(name))
    filters_e = encode_layer_siamese(filters_d, depth , 512, [3,3],False, "basic_5_{}".format(name))
    return filters_e