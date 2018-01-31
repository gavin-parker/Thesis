import tensorflow as tf

tf.app.flags.DEFINE_float('learning-rate', 1e-4, 'Number of examples to run. (default: %(default)d)')
tf.app.flags.DEFINE_integer('max-epochs', 200, 'Number of epochs to run. (default: %(default)d)')

FLAGS = tf.app.flags.FLAGS


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


def encode_layer(x, count, size, stride):
    conv = conv2d_extraction(x, count, size, stride)
    conv_bn = tf.layers.batch_normalization(conv)
    return conv_bn


def decode_layer(x, count, size, stride):
    deconv = conv2d_reconstruction(x, count, size, stride)
    deconv_bn = tf.layers.batch_normalization(deconv)
    return deconv_bn


def pool(x, strides=(2, 2)):
    return tf.layers.average_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=strides,
        padding='same')


class Model:
    def __init__(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.images = tf.placeholder(tf.float32, [1, None, None, 3])
        self.normals = tf.placeholder(tf.float32, [1, None, None, 3])
        self.envmaps = tf.placeholder(tf.float32, [1, None, None, 3])
        self.images = tf.map_fn(lambda x: tf.image.resize_image_with_crop_or_pad(x, 128, 128), self.images)
        self.normals = tf.map_fn(lambda x: tf.image.resize_image_with_crop_or_pad(x, 128, 128), self.normals)
        self.envmaps = tf.map_fn(lambda x: tf.image.resize_image_with_crop_or_pad(x, 512, 1024), self.envmaps)

        self.images = tf.map_fn(lambda x: tf.image.per_image_standardization(x), self.images)
        self.normals = tf.map_fn(lambda x: tf.image.per_image_standardization(x), self.normals)
        self.envmaps = tf.map_fn(lambda x: tf.image.per_image_standardization(x), self.envmaps)

        self.pred = self.generate(self.images, self.normals)
        self.pred = tf.map_fn(lambda x: tf.image.resize_image_with_crop_or_pad(x, 512, 1024), self.pred)

        self.pred = tf.where(tf.is_nan(self.pred), tf.zeros_like(self.pred), self.pred)
        self.pred = tf.where(tf.is_inf(self.pred), tf.zeros_like(self.pred), self.pred)
        self.envmaps = tf.where(tf.is_nan(self.envmaps), tf.zeros_like(self.envmaps), self.envmaps)
        self.envmaps = tf.where(tf.is_inf(self.envmaps), tf.zeros_like(self.envmaps), self.envmaps)

        self.loss = tf.reduce_mean(tf.square(self.envmaps - self.pred))

        self.img_out_summary = tf.summary.image('generated envmaps', self.pred, max_outputs=1)
        self.img_in_summary = tf.summary.image('trained renders', self.images, max_outputs=1)
        self.img_target_summary = tf.summary.image('trained renders', self.envmaps, max_outputs=1)
        self.img_normal_summary = tf.summary.image('normals', self.normals, max_outputs=1)
        self.img_summary = tf.summary.merge(
            [self.img_out_summary, self.img_in_summary, self.img_target_summary, self.img_normal_summary])
        self.train_writer = tf.summary.FileWriter("train_summaries", self.sess.graph)
        self.saver = tf.train.Saver()
        return

    # Convolutional graph definition for render -> envmap
    def generate(self, x_image, x_normal):
        x_input = tf.concat([x_image, x_normal], 1)
        encode_1 = encode_layer(x_input, 64, (5, 5), (2, 2))
        encode_2 = encode_layer(encode_1, 128, (3, 3), (2, 2))
        encode_3 = encode_layer(encode_2, 256, (3, 3), (2, 2))
        encode_4 = encode_layer(encode_3, 512, (3, 3), (2, 2))
        encode_5 = encode_layer(encode_4, 512, (3, 3), (2, 2))
        encode_6 = encode_layer(encode_5, 512, (3, 3), (2, 2))
        encode_7 = encode_layer(encode_6, 512, (3, 3), (2, 2))
        encode_8 = encode_layer(encode_7, 512, (3, 3), (2, 2))

        decode_1 = decode_layer(encode_8, 512, (3, 3), (1, 2))
        decode_2 = decode_layer(decode_1, 512, (3, 3), (2, 2))
        decode_3 = decode_layer(decode_2, 512, (3, 3), (2, 2))
        decode_4 = decode_layer(decode_3, 512, (3, 3), (2, 2))
        decode_5 = decode_layer(decode_4, 512, (3, 3), (2, 2))
        decode_6 = decode_layer(decode_5, 512, (3, 3), (2, 2))
        decode_7 = decode_layer(decode_6, 512, (3, 3), (2, 2))
        decode_8 = decode_layer(decode_7, 256, (3, 3), (2, 2))
        decode_9 = decode_layer(decode_8, 128, (3, 3), (2, 2))
        decode_10 = decode_layer(decode_9, 128, (3, 3), (2, 2))
        decode_11 = decode_layer(decode_10, 3, (3, 3), (1, 1))

        return decode_11

    def train(self, dataset):
        self.train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        # generate batches and run graph
        for epoch in range(0, FLAGS.max_epochs):
            total_err = 0
            batch_count = 0
            for image, normal, envmap in dataset.generate_batches():
                _, err, summary = self.sess.run([self.train_op, self.loss, self.img_summary],
                                                feed_dict={self.envmaps: envmap, self.images: image, self.normals: normal})
                total_err += err
                batch_count += 1
                self.train_writer.add_summary(summary, epoch)
            self.train_writer.flush()
            print(total_err / batch_count)
        self.train_writer.close()
        self.sess.close()
