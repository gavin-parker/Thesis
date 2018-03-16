import tensorflow as tf

"""Takes a 256x256 image and reduces to 8x8xN, using shared weights"""

def siamese_encode(input, reuse=False):
    with tf.name_scope("siamese_encode"):
        with tf.variable_scope("siamese_1") as scope:
            input = tf.layers.conv2d(input, 64, [3, 3], activation=tf.nn.relu, padding='SAME',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   reuse=reuse)
            input = tf.layers.max_pooling2d(input, pool_size=[2, 2], strides=[2, 2], padding='SAME')
        with tf.variable_scope("siamese_2") as scope:
            input = tf.layers.conv2d(input, 128, [3, 3], activation=tf.nn.relu, padding='SAME',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   reuse=reuse)
            input = tf.layers.max_pooling2d(input, pool_size=[2, 2], strides=[2, 2], padding='SAME')
        with tf.variable_scope("siamese_3") as scope:
            input = tf.layers.conv2d(input, 256, [3, 3], activation=tf.nn.relu, padding='SAME',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   reuse=reuse)
            input = tf.layers.max_pooling2d(input, pool_size=[2, 2], strides=[2, 2], padding='SAME')
        with tf.variable_scope("siamese_4") as scope:
            input = tf.layers.conv2d(input, 512, [3, 3], activation=tf.nn.relu, padding='SAME',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   reuse=reuse)
            input = tf.layers.max_pooling2d(input, pool_size=[2, 2], strides=[2, 2], padding='SAME')
        with tf.variable_scope("siamese_5") as scope:
            input = tf.layers.conv2d(input, 512, [3, 3], activation=tf.nn.relu, padding='SAME',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   reuse=reuse)
            input = tf.layers.max_pooling2d(input, pool_size=[2, 2], strides=[2, 2], padding='SAME')
        with tf.variable_scope("siamese_6") as scope:
            input = tf.layers.conv2d(input, 512, [3, 3], activation=tf.nn.relu, padding='SAME',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   reuse=reuse)
            input = tf.layers.max_pooling2d(input, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    return input