import tensorflow as tf



def conv2d_extraction(x, filters, size, strides=[1, 1]):
    return tf.layers.conv2d(inputs=x,
                            filters=filters,
                            strides=strides,
                            kernel_initializer=tf.random_uniform_initializer(-0.05, 0.05),
                            padding='SAME',
                            #activation=tf.nn.relu,
                            kernel_size=size)


def conv2d_reconstruction(x, filters, size, strides):
    return tf.layers.conv2d_transpose(inputs=x,
                                      filters=filters,
                                      kernel_size=size,
                                      strides=strides,
                                      padding='SAME',
                                      #activation=tf.nn.relu
                                        )


def dense(x, units):
    return tf.layers.dense(inputs=x,
                           units=units,
                           kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                           )


def encode_layer(x, count, size, stride, convolutions=1):
    for i in range(0,convolutions):
        x = conv2d_extraction(x, count, size, [1, 1])
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
    conv_pool = pool(x, stride)
    return conv_pool


def decode_layer(x, count, size, stride, convolutions=0, dropout=1.0):
    deconv = conv2d_reconstruction(x, count, size, stride)
    for i in range(0,convolutions):
        deconv = conv2d_extraction(deconv, count, size, [1, 1])
        deconv = tf.layers.batch_normalization(deconv)
        deconv = tf.nn.relu(deconv)
    return deconv


def pool(x, strides=(2, 2)):
    return tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=strides,
        padding='same')
