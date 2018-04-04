import tensorflow as tf
from reflectance_ops import Reflectance
import os
import preprocessing_ops as preprocessing
import layers as ed
import time
from params import FLAGS

""" Convolutional-Deconvolutional model for extracting reflectance maps from input images with normals.
    Extracts sparse reflectance maps, with the CNN performing data interpolation"""

"""Defines the graph and provides a template training function"""


class Model:
    appearance = tf.placeholder(tf.float32, [FLAGS.batch_size, 128, 128, 3])
    orientation = tf.placeholder(tf.float32, [FLAGS.batch_size, 128, 128, 3])
    gt = tf.placeholder(tf.float32, [FLAGS.batch_size, 128, 128, 3])
    sparse_rm = tf.placeholder(tf.float32, [FLAGS.batch_size, 128, 128, 3])
    train_path = "synthetic/synthetic/train"
    input_files = preprocessing.image_stream("{}/radiance/*.jpg".format(train_path))
    normal_files = preprocessing.image_stream("{}/normal/*.png".format(train_path))
    gt_files = preprocessing.image_stream("{}/lit/*.png".format(train_path))
    norm_sphere = tf.image.decode_image(tf.read_file("synthetic/normal_sphere.png"), channels=3)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                               1, 0.95, staircase=False)

    def __init__(self):

        inputs = self.input_files.map(preprocessing.preprocess_color, num_parallel_calls=4)
        normals = self.normal_files.map(preprocessing.preprocess_orientation, num_parallel_calls=4)
        gts = self.gt_files.map(preprocessing.preprocess_gt, num_parallel_calls=4)
        self.sphere = preprocessing.get_norm_sphere(self.norm_sphere, FLAGS.batch_size)
        self.batch_sphere = preprocessing.unit_norm(self.sphere[:, :, :, :3], channels=3)
        dataset = tf.data.Dataset.zip((inputs, normals, gts)).repeat().batch(FLAGS.batch_size).shuffle(128).prefetch(buffer_size=4)
        iterator = dataset.make_one_shot_iterator()
        input_batch = iterator.get_next()
        self.zero_mask = preprocessing.zero_mask(input_batch[2])
        prediction, sparse_rm = self.inference(input_batch)
        self.loss_calculation(prediction, input_batch[2])
        self.train_op = self.optimize()
        self.summaries = self.summary(input_batch[0], input_batch[1], input_batch[2], sparse_rm, prediction,
                                      self.sphere)
        return

    """Calculate a prediction RM and intermediary sparse RM"""

    def inference(self, input_batch):
        (appearance, orientation, gt) = input_batch
        normal_sphere_flat = tf.reshape(self.batch_sphere, [FLAGS.batch_size, -1, 3])
        with tf.device('/gpu:0'):
            sparse_rm = tf.map_fn(Reflectance.online_reflectance,
                                  (appearance, orientation, normal_sphere_flat), dtype=tf.float16, name="reflectance")
            sparse_rm = tf.cast(tf.reshape(sparse_rm, [FLAGS.batch_size, 128, 128, 3]), dtype=tf.float32, name="recast")

            indirect, feature_maps = self.encode(sparse_rm, 512)
            prediction = self.decode(indirect, feature_maps, FLAGS.batch_size)
        return tf.multiply(prediction, self.zero_mask), sparse_rm

    """Calculate the l2 norm loss between the prediction and ground truth"""

    def loss_calculation(self, prediction, gt):
        self.loss = tf.sqrt(tf.reduce_sum(tf.square(gt - prediction)), name="l2_norm")

    """Use Adam Optimizer to minimize loss"""

    def optimize(self):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def encode(self, image, out_size):
        encode_1 = ed.encode_layer(image, 3, (11, 11), (2, 2))
        encode_2 = ed.encode_layer(encode_1, 64, (7, 7), (2, 2))
        encode_3 = ed.encode_layer(encode_2, 128, (3, 3), (2, 2))
        encode_4 = ed.encode_layer(encode_3, 512, (8, 8), (16, 16))

        full_1 = tf.contrib.layers.fully_connected(encode_4, out_size)
        return full_1, [encode_4, encode_3, encode_2]

    def decode(self, full, feature_maps, batch_size):
        full_2 = tf.concat([full, feature_maps[0]], axis=-1)
        if feature_maps:
            decode_1 = ed.decode_layer(full_2, 512, (3, 3), (8, 8))  # 2,8,8,512
            fm_1 = tf.reshape(feature_maps[0], [batch_size, 8, 8, -1])
            decode_2 = ed.decode_layer(tf.concat([decode_1, fm_1], axis=-1), 256, (3, 3), (2, 2))
            decode_3 = ed.decode_layer(tf.concat([decode_2, feature_maps[1]], axis=-1), 128, (3, 3), (2, 2))
            decode_4 = ed.decode_layer(tf.concat([decode_3, feature_maps[2]], axis=-1), 3, (3, 3), (1, 1))
        return decode_4

    """Create tensorboard summaries of images and loss"""

    def summary(self, appearance, orientation, gt, sparse_rm, pred, sphere):
        img_out_summary = tf.summary.image('generated reflectance', pred, max_outputs=1)
        norm_sphere_summary = tf.summary.image('Gauss Sphere', sphere, max_outputs=1)
        sparse_rm_summary = tf.summary.image('sparse reflectance', sparse_rm, max_outputs=1)
        img_in_summary = tf.summary.image('training input', appearance, max_outputs=1)
        img_normal_summary = tf.summary.image('normal input', orientation, max_outputs=1)
        img_gt_summary = tf.summary.image('Ground Truth', gt, max_outputs=1)
        loss_summary = tf.summary.scalar("Loss", self.loss)
        img_summary = tf.summary.merge(
            [img_out_summary, img_in_summary, img_gt_summary, img_normal_summary, sparse_rm_summary,
             norm_sphere_summary])
        return loss_summary, img_summary

    """Train the model with the settings provided in FLAGS"""

    def train(self, sess=None):

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        if sess == None:
            sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        options, run_metadata = None, None
        if FLAGS.debug:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        tf.train.start_queue_runners(sess=sess)
        train_writer = tf.summary.FileWriter("synthetic_results_pool/{}".format(time.strftime("%H:%M:%S")), sess.graph)

        saver = tf.train.Saver()
        epoch_size = 50000 / FLAGS.max_epochs
        # generate batches and run graph
        for epoch in range(0, FLAGS.max_epochs):
            for i in range(0, epoch_size):
                sess.run([self.train_op], feed_dict={self.global_step: epoch}, options=options,
                         run_metadata=run_metadata)
                if i % 10 == 0:
                    err, (summary, loss_summary) = sess.run(
                        [self.loss, self.summaries], options=options,
                        run_metadata=run_metadata)
                    train_writer.add_summary(summary, epoch * epoch_size + i)
                    train_writer.add_summary(loss_summary, epoch * epoch_size + i)
                    saver.save(sess, os.path.join("reflectance_graph", 'model'), global_step=epoch)
                    if FLAGS.debug:
                        train_writer.add_run_metadata(run_metadata, "step{}".format(epoch * epoch_size + i),
                                                      global_step=None)
                    train_writer.flush()
                    print(err)
        print("finished")
        train_writer.close()
        sess.close()
