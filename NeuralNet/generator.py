import tensorflow as tf
tf.app.flags.DEFINE_float('learning-rate', 1e-5, 'Number of examples to run. (default: %(default)d)')
tf.app.flags.DEFINE_integer('max-epochs', 50, 'Number of epochs to run. (default: %(default)d)')

FLAGS = tf.app.flags.FLAGS


def conv2d_extraction(x, filters, size, strides=[1, 1]):
    return tf.layers.conv2d(inputs=x,
                            filters=filters,
                            strides=strides,
                            kernel_initializer=tf.random_uniform_initializer(-0.05, 0.05),
                            padding='SAME',
                            activation=tf.nn.relu,
                            kernel_size=size)


def conv2d_reconstruction(x, strides):
    return tf.layers.conv2d_transpose(inputs=x,
                                      filters=3,
                                      kernel_size=[5, 5],
                                      strides=strides,
                                      padding='SAME',
                                      activation=tf.nn.relu)


def dense(x, units):
    return tf.layers.dense(inputs=x,
                           units=units,
                           kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                           )



def pool(x):
    return tf.layers.average_pooling2d(
        inputs=x,
        pool_size=[3, 3],
        strides=2,
        padding='same')


class Generator:
    def __init__(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        envmap_size, input_size = (1536, 256), (1080, 1920)
        self.envmaps = tf.placeholder(tf.float32, [1, envmap_size[0], envmap_size[1], 3])
        self.renders = tf.placeholder(tf.float32, [1, input_size[0], input_size[1], 3])
        self.probes = tf.map_fn(lambda x: tf.image.resize_image_with_crop_or_pad(x, 256, 256), self.renders)
        # self.envmaps = tf.map_fn(lambda x: tf.image.per_image_standardization(x), self.envmaps)
        # self.renders = tf.map_fn(lambda x: tf.image.per_image_standardization(x), self.renders)
        self.pred = self.generate(self.probes)
        self.resized_pred = tf.image.resize_images(self.pred, [1536, 256])
        self.loss = tf.reduce_mean(tf.square(self.envmaps - self.resized_pred))
        self.img_out_summary = tf.summary.image('generated envmaps', self.pred)
        self.img_out_summary_r = tf.summary.image('generated resized envmaps', self.resized_pred)
        self.img_in_summary = tf.summary.image('trained renders', self.probes)
        self.img_target_summary = tf.summary.image('trained renders', self.envmaps)
        self.img_summary = tf.summary.merge([self.img_out_summary, self.img_out_summary_r, self.img_in_summary, self.img_target_summary])
        self.train_writer = tf.summary.FileWriter("train_summaries", self.sess.graph)
        self.saver = tf.train.Saver()
        return

    # Convolutional graph definition for render -> envmap
    def generate(self, x_image):
        conv1 = conv2d_extraction(x_image, 128, [11, 11], [1, 1])
        #pool1 = pool(conv1)
        conv2 = conv2d_extraction(conv1, 32, [7, 7], [1, 1])
        #pool2 = pool(conv2)
        conv3 = conv2d_extraction(conv2, 16, [3, 3], [1, 1])
        conv3_bn = tf.layers.batch_normalization(conv3)

        flat = tf.reshape(conv3_bn, [8, 8, -1])
        fullc1 = dense(flat, 16384)
        fullc2 = dense(fullc1, 16384)
        reshaped = tf.reshape(fullc2, [1, 256, 256, 16])

        deconv1 = conv2d_reconstruction(reshaped, [1, 1])
        #deconv2 = conv2d_reconstruction(deconv1, [2, 2])
        #deconv3 = conv2d_reconstruction(deconv2, [6, 1])
        return deconv1

    def train(self, dataset):
        self.train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        # generate batches and run graph
        for epoch in range(0, FLAGS.max_epochs):
            for sample, envmap in dataset.generate_batches(testing=False):
                _, err, summary = self.sess.run([self.train_op, self.loss, self.img_summary],
                                                feed_dict={self.envmaps: envmap, self.renders: sample})
                if epoch % 5 == 0:
                    self.train_writer.add_summary(summary, epoch)
                    self.train_writer.flush()
        self.train_writer.close()
        self.sess.close()
