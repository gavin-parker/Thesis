import tensorflow as tf


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class Discriminator:
    def __init__(self):
        self.kernel_initializer = tf.random_uniform_initializer(-0.05, 0.05)
        self.kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
        self.w_fc1 = 32
        self.w_fc2 = 16
        return

    # Take an input rendered image and classify as real or fake
    def discriminate(self, x_image, output=2):
        b_conv1 = conv2d(x_image, 64)
        h_conv1 = tf.nn.relu(conv2d(x_image, 64) + b_conv1)
        h_pool1 = avg_pool_2x2(h_conv1)

        b_conv2 = conv2d(h_pool1, 64)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, 64) + b_conv2)
        h_pool2 = avg_pool_2x2(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 16])
        h_fullc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.w_fc1))

        logits = tf.layers.dense(inputs=h_fullc1,
                                 units=output,
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                                 name='fc1',
                                 )
        return logits
