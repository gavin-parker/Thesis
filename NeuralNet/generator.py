import tensorflow as tf

tf.app.flags.DEFINE_float('learning-rate', 1e-5, 'Number of examples to run. (default: %(default)d)')
tf.app.flags.DEFINE_integer('max-epochs', 50, 'Number of epochs to run. (default: %(default)d)')

FLAGS = tf.app.flags.FLAGS


def conv2d_extraction(x, filters, size, strides=[1,1]):
    return tf.layers.conv2d(inputs=x,
                            filters=filters,
                            strides=strides,
                            kernel_initializer=tf.random_uniform_initializer(-0.05, 0.05),
                            padding='SAME',
                            activation=tf.nn.relu,
                            kernel_size=size)


def conv2d_reconstruction(x):
    return tf.layers.conv2d_transpose(inputs=x,
                                      filters=3,
                                      kernel_size=[5, 5],
                                      strides=[6, 1],
                                      padding='SAME',
                                      activation=tf.nn.relu)


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
        self.loss = tf.reduce_mean(tf.square(self.envmaps - self.pred))
        self.img_out_summary =  tf.summary.image('generated envmaps', self.pred)
        self.img_in_summary =  tf.summary.image('trained renders', self.probes)
        self.img_target_summary =  tf.summary.image('trained renders', self.envmaps)
        self.img_summary = tf.summary.merge([self.img_out_summary, self.img_in_summary, self.img_target_summary])
        self.train_writer = tf.summary.FileWriter("train_summaries", self.sess.graph)
        self.saver = tf.train.Saver()
        return

    # Convolutional graph definition for render -> envmap
    def generate(self, x_image):
        conv1 = conv2d_extraction(x_image, 64, [9, 9], [1, 1])
        pool1 = tf.layers.average_pooling2d(
            inputs=conv1,
            pool_size=[3, 3],
            strides=1,
            padding='same',
            name='pool1'
        )
        conv2 = conv2d_extraction(pool1, 32, [5, 5], [1, 1])
        conv3 = conv2d_reconstruction(conv2)
        return conv3

    def train(self, dataset):
        self.train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        # generate batches and run graph
        for epoch in range(0, FLAGS.max_epochs):
            for sample, envmap in dataset.generate_batches(testing=False):
                _, err, summary = self.sess.run([self.train_op, self.loss, self.img_summary], feed_dict={self.envmaps: envmap, self.renders: sample})
                if epoch % 5 == 0:
                    self.train_writer.add_summary(summary, epoch)
                    self.train_writer.flush()
                print(err)
        self.train_writer.close()
        self.sess.close()