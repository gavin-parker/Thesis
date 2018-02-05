import tensorflow as tf
from reflectance_ops import offline_reflectance, online_reflectance
import math
from reflectance_dataset import ReflectanceDataset
import numpy as np
import glob
from PIL import Image
tf.app.flags.DEFINE_float('learning-rate', 1e-4, 'Number of examples to run. (default: %(default)d)')
tf.app.flags.DEFINE_integer('max-epochs', 200, 'Number of epochs to run. (default: %(default)d)')
tf.app.flags.DEFINE_integer('batch-size', 2, 'Batch Size. (default: %(default)d)')

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


def format_image(image, in_shape, out_size):
    image = tf.reshape(image, in_shape)
    image = tf.image.resize_images(image, [out_size,out_size])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image #tf.map_fn(tf.image.per_image_standardization, image)


class Model:
    appearance = tf.placeholder(tf.float32, [FLAGS.batch_size, 128, 128, 3])
    orientation = tf.placeholder(tf.float32, [FLAGS.batch_size, 128, 128, 3])
    gt = tf.placeholder(tf.float32, [FLAGS.batch_size, 128, 128, 3])
    sparse_rm = tf.placeholder(tf.float32, [FLAGS.batch_size, 128, 128, 3])
    train_path = "synthetic/synthetic/train"
    input_files = tf.contrib.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(sorted(glob.glob("{}/radiance/*.jpg".format(train_path)))))
    normal_files = tf.contrib.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(sorted(glob.glob("{}/normal/*.png".format(train_path)))))
    gt_files = tf.contrib.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(sorted(glob.glob("{}/lit/*.png".format(train_path)))))

    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        self.sess = tf.Session(config=config)
        inputs = self.input_files.map(lambda filename: tf.image.decode_image(tf.read_file(filename), channels=3))
        normals = self.normal_files.map(lambda filename: tf.image.decode_image(tf.read_file(filename), channels=4))
        gts = self.gt_files.map(lambda filename: tf.image.decode_image(tf.read_file(filename), channels=3))

        dataset = tf.data.Dataset.zip((inputs, normals, gts)).repeat().batch(FLAGS.batch_size)
        iterator = dataset.make_one_shot_iterator()
        (appearance, orientation, gt) = iterator.get_next()
        self.appearance = format_image(appearance, [FLAGS.batch_size,256,256,3], 128)
        self.orientation = format_image(orientation, [FLAGS.batch_size,256,256,4], 128)
        alpha = tf.clip_by_value(self.orientation[:,:,:,3], 0, 1)
        orientation_mask = tf.stack([alpha,alpha,alpha,alpha], axis=-1)
        self.orientation = tf.multiply(self.orientation, orientation_mask)[:,:,:,:3]
        self.sparse_rm = tf.map_fn(online_reflectance, (self.appearance, self.orientation), dtype=tf.float32)
        self.sparse_rm = tf.multiply(self.sparse_rm, orientation_mask[:,:,:,:3])
        self.gt = format_image(gt, [FLAGS.batch_size,256,256,3], 32)

        pred = self.generate(self.appearance, self.sparse_rm)
        self.loss = tf.reduce_mean(tf.square(self.gt - pred))

        img_out_summary = tf.summary.image('generated reflectance', pred, max_outputs=1)
        sparse_rm_summary = tf.summary.image('sparse reflectance', self.sparse_rm, max_outputs=1)
        img_in_summary = tf.summary.image('training input', self.appearance, max_outputs=1)
        img_normal_summary = tf.summary.image('normal input', self.orientation, max_outputs=1)
        img_gt_summary = tf.summary.image('Ground Truth', self.gt, max_outputs=1)
        self.img_summary = tf.summary.merge(
            [img_out_summary, img_in_summary, img_gt_summary, img_normal_summary, sparse_rm_summary])
        self.train_writer = tf.summary.FileWriter("train_summaries", self.sess.graph)
        self.saver = tf.train.Saver()
        return

    def encode(self, image, out_size):
        encode_1 = encode_layer(image, 3, (11, 11), (2, 2))
        encode_2 = encode_layer(encode_1, 64, (7, 7), (2, 2))
        encode_3 = encode_layer(encode_2, 128, (3, 3), (2, 2))
        encode_4 = encode_layer(encode_3, 512, (8, 8), (16, 16))

        full_1 = tf.contrib.layers.fully_connected(encode_4, out_size)
        return full_1

    def decode(self, full, size):
        full_2 = tf.contrib.layers.fully_connected(full, size)

        decode_1 = decode_layer(full_2, 512, (3, 3), (4, 4))
        decode_2 = decode_layer(decode_1, 256, (3, 3), (2, 2))
        decode_3 = decode_layer(decode_2, 128, (3, 3), (2, 2))
        decode_4 = decode_layer(decode_3, 3, (3, 3), (2, 2))

        return decode_4

    # Convolutional graph definition for render -> envmap
    def generate(self, image, sparse_rm):
        direct = self.encode(image, 512)
        indirect = self.encode(sparse_rm, 512)
        full = tf.concat([direct, indirect], axis=-1)
        full_1 = tf.contrib.layers.fully_connected(full, 512)
        return self.decode(full_1, 512)

    def train(self):
        self.train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners(sess=self.sess)
        run_metadata = tf.RunMetadata()
        print(run_metadata)
        np.set_printoptions(threshold=5000)
        # generate batches and run graph
        for epoch in range(0, FLAGS.max_epochs):
            total_err = 0.0
            summary = {}
            for i in range(0, 10):
                #test = self.sess.run(self.orientation_4d)
                #print(test)
                _, err, summary = self.sess.run([self.train_op, self.loss, self.img_summary])
                total_err += err
            self.train_writer.add_summary(summary, epoch)
            self.train_writer.flush()
            print(total_err/2.0)
        print("finished")
        self.train_writer.close()
        self.sess.close()
