import tensorflow as tf
from reflectance_dataset import ReflectanceDataset
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

def image_queue(image_reader, file_names):
    input_queue = tf.train.string_input_producer(file_names)
    _, input_file = image_reader.read(input_queue)
    input_image = tf.image.decode_png(input_file)
    input_image.set_shape((256, 256, 3))
    return input_image


class Model:
    image = tf.placeholder(tf.float32, [ 256, 256, 3])
    normals = tf.placeholder(tf.float32, [ 256, 256, 3])
    gts = tf.placeholder(tf.float32, [ 256, 256, 3])
    train_path = "synthetic/synthetic/train"
    input_files = tf.train.match_filenames_once("{}/radiance/*.png".format(train_path))
    normal_files = tf.train.match_filenames_once("{}/normal/*.png".format(train_path))
    gt_files = tf.train.match_filenames_once("{}/lit/*.png".format(train_path))

    def __init__(self):
        self.sess = tf.Session()
        self.image_reader = tf.WholeFileReader()
        self.input_image = image_queue(self.image_reader, self.input_files)
        self.normal_image = image_queue(self.image_reader, self.normal_files)
        self.gt_image = image_queue(self.image_reader, self.gt_files)

        self.images = tf.to_float(tf.train.shuffle_batch([self.input_image, self.normal_image, self.gt_image],
                                        batch_size=1,
                                        num_threads=4,
                                        capacity=100,
                                        min_after_dequeue=0))

        tf.Print(self.images, [self.images], "got images")
        self.pred = self.generate(self.images[0], self.images[1])
        self.loss = tf.reduce_mean(tf.square(self.images[2] - self.pred))

        self.img_out_summary = tf.summary.image('generated reflectance', self.pred, max_outputs=1)
        self.img_in_summary = tf.summary.image('training input', self.images[0], max_outputs=1)
        self.img_normal_summary = tf.summary.image('normal input', self.images[1], max_outputs=1)
        self.img_gt_summary = tf.summary.image('Ground Truth', self.images[2], max_outputs=1)
        self.img_summary = tf.summary.merge(
            [self.img_out_summary, self.img_in_summary,self.img_gt_summary, self.img_normal_summary])
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
        encode_8 = encode_layer(encode_7, 512, (3, 3), (4, 2))

        decode_1 = decode_layer(encode_8, 512, (3, 3), (2, 2))
        decode_2 = decode_layer(decode_1, 512, (3, 3), (2, 2))
        decode_3 = decode_layer(decode_2, 512, (3, 3), (2, 2))
        decode_4 = decode_layer(decode_3, 512, (3, 3), (2, 2))
        decode_5 = decode_layer(decode_4, 512, (3, 3), (2, 2))
        decode_6 = decode_layer(decode_5, 512, (3, 3), (2, 2))
        decode_7 = decode_layer(decode_6, 512, (3, 3), (2, 2))
        decode_8 = decode_layer(decode_7, 3, (3, 3), (2, 2))

        return decode_8

    def train(self):
        self.train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners(sess=self.sess)

        # generate batches and run graph
        for epoch in range(0, FLAGS.max_epochs):
            _, err, summary = self.sess.run([self.train_op, self.loss, self.img_summary])
            print(err)
        self.train_writer.close()
        self.sess.close()
