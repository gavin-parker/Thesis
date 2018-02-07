import tensorflow as tf
from reflectance_ops import Reflectance
import numpy as np
import glob
from preprocessing_ops import Preprocessor

tf.app.flags.DEFINE_float('learning-rate', 1e-4, 'Learning Rate. (default: %(default)d)')
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
    norm_sphere = tf.image.decode_image(tf.read_file("synthetic/normal_sphere.png"), channels=3)

    def __init__(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        self.sess = tf.Session(config=config)
        inputs = self.input_files.map(Preprocessor.preprocess_color, num_parallel_calls=4)
        normals = self.normal_files.map(Preprocessor.preprocess_orientation, num_parallel_calls=4)
        gts = self.gt_files.map(Preprocessor.preprocess_gt, num_parallel_calls=4)
        sphere = Preprocessor.get_norm_sphere(self.norm_sphere, FLAGS.batch_size)
        self.batch_sphere = Preprocessor.unit_norm(sphere[:, :, :, :3], channels=3)
        dataset = tf.data.Dataset.zip((inputs, normals, gts)).repeat().batch(FLAGS.batch_size).prefetch(buffer_size=4)
        iterator = dataset.make_one_shot_iterator()
        (appearance, orientation, gt) = iterator.get_next()
        normal_sphere_flat = tf.reshape(self.batch_sphere, [FLAGS.batch_size, -1, 3])

        with tf.device('/gpu:0'):
            sparse_rm = tf.map_fn(Reflectance.online_reflectance,
                                       (appearance, orientation, normal_sphere_flat), dtype=tf.float16, name="reflectance")
            sparse_rm = tf.cast(tf.reshape(sparse_rm, [FLAGS.batch_size, 128, 128, 3]), dtype=tf.float32, name="recast")
            pred = self.generate(sparse_rm)
            self.loss = tf.sqrt(tf.reduce_sum(tf.square(gt - pred)))
        self.summary(appearance, orientation, gt, sparse_rm, pred, sphere)

        return

    def encode(self, image, out_size):
        encode_1 = encode_layer(image, 3, (11, 11), (2, 2))
        encode_2 = encode_layer(encode_1, 64, (7, 7), (2, 2))
        encode_3 = encode_layer(encode_2, 128, (3, 3), (2, 2))
        encode_4 = encode_layer(encode_3, 512, (8, 8), (16, 16))

        full_1 = tf.contrib.layers.fully_connected(encode_4, out_size)
        return full_1, [encode_4, encode_3, encode_2]

    def decode(self, full, size, feature_maps):
        full_2 = tf.concat([full, feature_maps[0]], axis=-1)
        if feature_maps:
            decode_1 = decode_layer(full_2, 512, (3, 3), (8, 8))  # 2,8,8,512
            fm_1 = tf.reshape(feature_maps[0], [FLAGS.batch_size, 8, 8, -1])
            decode_1 = tf.concat([decode_1, fm_1], axis=-1)
            decode_2 = decode_layer(decode_1, 256, (3, 3), (2, 2))
            decode_2 = tf.concat([decode_2, feature_maps[1]], axis=-1)
            decode_3 = decode_layer(decode_2, 128, (3, 3), (2, 2))
            decode_3 = tf.concat([decode_3, feature_maps[2]], axis=-1)
            decode_4 = decode_layer(decode_3, 3, (3, 3), (1, 1))
        else:
            decode_1 = decode_layer(full_2, 512, (3, 3), (2, 2))
            decode_2 = decode_layer(decode_1, 256, (3, 3), (2, 2))
            decode_3 = decode_layer(decode_2, 128, (3, 3), (2, 2))
            decode_4 = decode_layer(decode_3, 3, (3, 3), (2, 2))

        return decode_4

    # Convolutional graph definition for render -> envmap
    def generate(self, sparse_rm):
        indirect, feature_maps = self.encode(sparse_rm, 512)
        return self.decode(indirect, 512, feature_maps)

    def summary(self, appearance, orientation, gt, sparse_rm, pred, sphere):
        img_out_summary = tf.summary.image('generated reflectance', pred, max_outputs=1)
        norm_sphere_summary = tf.summary.image('Gauss Sphere', sphere, max_outputs=1)
        sparse_rm_summary = tf.summary.image('sparse reflectance', sparse_rm, max_outputs=1)
        img_in_summary = tf.summary.image('training input', appearance, max_outputs=1)
        img_normal_summary = tf.summary.image('normal input', orientation, max_outputs=1)
        img_gt_summary = tf.summary.image('Ground Truth', gt, max_outputs=1)
        self.loss_summary = tf.summary.scalar("Loss", self.loss)
        self.img_summary = tf.summary.merge(
            [img_out_summary, img_in_summary, img_gt_summary, img_normal_summary, sparse_rm_summary,
             norm_sphere_summary])
        self.train_writer = tf.summary.FileWriter("train_summaries_c", self.sess.graph)
        self.saver = tf.train.Saver()

    def train(self):
        self.train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()
        tf.train.start_queue_runners(sess=self.sess)
        # generate batches and run graph
        for epoch in range(0, FLAGS.max_epochs):
            for i in range(0, 50000/FLAGS.batch_size):
                self.sess.run([self.train_op], options=options, run_metadata=self.run_metadata)
                if i % 1 == 0:
                    err, summary, loss_summary = self.sess.run(
                        [ self.loss, self.img_summary, self.loss_summary], options=options,
                        run_metadata=self.run_metadata)
                    self.train_writer.add_summary(summary, epoch)
                    self.train_writer.add_summary(loss_summary, epoch)
                    self.train_writer.add_run_metadata(self.run_metadata, "step{}".format(i), global_step=None)
                    self.train_writer.flush()
                    return
        print("finished")
        self.train_writer.close()
        self.sess.close()
