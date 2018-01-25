import tensorflow as tf

tf.app.flags.DEFINE_float('learning-rate', 0.0002, 'Number of examples to run. (default: %(default)d)')
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
    return tf.nn.relu(conv_bn)

def decode_layer(x, count, size, stride):
    deconv = conv2d_reconstruction(x, count, size, stride)
    deconv_bn = tf.layers.batch_normalization(deconv)
    return tf.nn.relu(deconv_bn)

def pool(x, strides=(2, 2)):
    return tf.layers.average_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=strides,
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
        self.img_summary = tf.summary.merge(
            [self.img_out_summary, self.img_out_summary_r, self.img_in_summary, self.img_target_summary])
        self.train_writer = tf.summary.FileWriter("train_summaries", self.sess.graph)
        self.saver = tf.train.Saver()
        return

    # Convolutional graph definition for render -> envmap
    def generate(self, x_image):
        encode_1 = encode_layer(x_image, 64, (5,5), (2,2) )
        encode_2 = encode_layer(encode_1, 128, (3,3), (2,2))
        encode_3 = encode_layer(encode_2, 256, (3,3), (2,2))
        encode_4 = encode_layer(encode_3, 512, (3,3), (2,2))
        encode_5 = encode_layer(encode_4, 512, (3,3), (2,2))
        encode_6 = encode_layer(encode_5, 512, (3,3), (2,2))
        encode_7 = encode_layer(encode_6, 512, (3,3), (2,2))
        encode_8 = encode_layer(encode_7, 512, (3,3), (2,2))

        decode_1 = decode_layer(encode_8, 512, (3,3), (6,1))
        decode_2= decode_layer(decode_1, 512, (3,3), (2,2))
        decode_3= decode_layer(decode_2, 512, (3,3), (2,2))
        decode_4= decode_layer(decode_3, 512, (3,3), (2,2))
        decode_5= decode_layer(decode_4, 256, (3,3), (2,2))
        decode_6= decode_layer(decode_5, 128, (3,3), (2,2))
        decode_7= decode_layer(decode_6, 64, (3,3), (2,2))
        decode_8= decode_layer(decode_7, 3, (3,3), (2,2))

        return decode_8

    def train(self, dataset):
        self.train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        # generate batches and run graph
        for epoch in range(0, FLAGS.max_epochs):
            total_err = 0
            batch_count = 0
            for sample, envmap in dataset.generate_batches(testing=False):
                _, err, summary = self.sess.run([self.train_op, self.loss, self.img_summary],
                                                feed_dict={self.envmaps: envmap, self.renders: sample})
                total_err += err
                batch_count += 1
                self.train_writer.add_summary(summary, epoch)
                self.train_writer.flush()
                print(total_err / batch_count)
        self.train_writer.close()
        self.sess.close()
