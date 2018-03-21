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
        with tf.variable_scope("siamese_4") as scope:
            filters_d = tf.layers.conv2d(filters_c, 512, [3, 3], activation=tf.nn.relu, padding='SAME',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   reuse=reuse)
            filters_d = tf.layers.max_pooling2d(filters_d, pool_size=[2, 2], strides=[2, 2], padding='SAME')
        with tf.variable_scope("siamese_5") as scope:
            filters_e = tf.layers.conv2d(filters_d, 512, [3, 3], activation=tf.nn.relu, padding='SAME',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   reuse=reuse)
            filters_e = tf.layers.max_pooling2d(filters_e, pool_size=[2, 2], strides=[2, 2], padding='SAME')
        with tf.variable_scope("siamese_6") as scope:
            filters_f = tf.layers.conv2d(filters_e, 512, [3, 3], activation=tf.nn.relu, padding='SAME',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   reuse=reuse)
            filters_f = tf.layers.max_pooling2d(filters_f, pool_size=[2, 2], strides=[2, 2], padding='SAME')
        with tf.variable_scope("siamese_6") as scope:
            filters_g = tf.layers.conv2d(filters_e, 512, [3, 3], activation=tf.nn.relu, padding='SAME',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   reuse=reuse)
            filters_g = tf.layers.max_pooling2d(filters_f, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    return [filters_a, filters_b, filters_c, filters_d, filters_e, filters_f, filters_g]