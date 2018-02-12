import tensorflow as tf


def encode(image, out_size):
    encode_1 = encode_layer(image, 3, (11, 11), (2, 2))
    encode_2 = encode_layer(encode_1, 64, (7, 7), (2, 2))
    encode_3 = encode_layer(encode_2, 128, (3, 3), (2, 2))
    encode_4 = encode_layer(encode_3, 512, (8, 8), (16, 16))

    full_1 = tf.contrib.layers.fully_connected(encode_4, out_size)
    return full_1, [encode_4, encode_3, encode_2]


def decode(full, feature_maps, batch_size):
    full_2 = tf.concat([full, feature_maps[0]], axis=-1)
    if feature_maps:
        decode_1 = decode_layer(full_2, 512, (3, 3), (8, 8))  # 2,8,8,512
        fm_1 = tf.reshape(feature_maps[0], [batch_size, 8, 8, -1])
        decode_1 = tf.concat([decode_1, fm_1], axis=-1)
        decode_2 = decode_layer(decode_1, 256, (3, 3), (2, 2))
        decode_2 = tf.concat([decode_2, feature_maps[1]], axis=-1)
        decode_3 = decode_layer(decode_2, 128, (3, 3), (2, 2))
        decode_3 = tf.concat([decode_3, feature_maps[2]], axis=-1)
        decode_4 = decode_layer(decode_3, 3, (3, 3), (1, 1))
    else:
        decode_1 = decode_layer(full_2, 512, (3, 3), (2, 2))
        decode_2 = decode_layer(decode_1, 256, (3, 3), (2, 2))
        decode_3 = decode_layer(decode_2, 128, (3, 3), (2, 2))
        decode_4 = decode_layer(decode_3, 3, (3, 3), (2, 2))
    return decode_4


def conv2d_extraction(x, filters, size, strides=[1, 1]):
    return tf.layers.conv2d(inputs=x,
                            filters=filters,
                            strides=strides,
                            kernel_initializer=tf.random_uniform_initializer(-0.05, 0.05),
                            padding='SAME',
                            activation=tf.nn.relu,
                            kernel_size=size)


def conv2d_reconstruction(x, filters, size, strides):
    return tf.layers.conv2d_transpose(inputs=x,
                                      filters=filters,
                                      kernel_size=size,
                                      strides=strides,
                                      padding='SAME',
                                      activation=tf.nn.relu)


def dense(x, units):
    return tf.layers.dense(inputs=x,
                           units=units,
                           kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                           )


def encode_layer(x, count, size, stride, convolutions=1):
    for i in range(0,convolutions):
        x = conv2d_extraction(x, count, size, [1, 1])
        x = tf.layers.batch_normalization(x)
    conv_pool = pool(x, stride)
    return conv_pool


def decode_layer(x, count, size, stride, convolutions=0):
    deconv = conv2d_reconstruction(x, count, size, stride)
    for i in range(0,convolutions):
        deconv = conv2d_extraction(deconv, count, size, [1, 1])
    deconv_bn = tf.layers.batch_normalization(deconv)
    return deconv_bn


def pool(x, strides=(2, 2)):
    return tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=strides,
        padding='same')
