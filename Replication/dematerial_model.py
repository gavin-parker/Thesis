import tensorflow as tf
import os
import preprocessing_ops as preprocessing
import encode_decode
import time
from params import FLAGS

""" Convolutional-Deconvolutional model for extracting reflectance maps from input images with normals.
    Extracts sparse reflectance maps, with the CNN performing data interpolation"""

"""Defines the graph and provides a template training function"""


class Model:
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                               1, 0.95, staircase=False)
    train_path = "/mnt/black/MultiNatIllum/data/multiple_materials_single_object/singlets/synthetic/train"
    bg_files = preprocessing.image_stream("{}/background/*.png".format(train_path))
    envmap_files = preprocessing.image_stream("{}/envmap_latlong/*.hdr".format(train_path))
    reflectance_files = preprocessing.image_stream("{}/reflectanceMap_latlong/*.png".format(train_path))

    def __init__(self):
        background = self.bg_files.map(
            lambda x: preprocessing.preprocess_color(x, input_shape=[128, 128, 3], double_precision=True),
            num_parallel_calls=4)
        envmap = self.envmap_files.map(lambda x: preprocessing.preprocess_hdr(x, 64, True), num_parallel_calls=4)
        reflectance = self.reflectance_files.map(lambda x: preprocessing.preprocess_color(x, [128, 128, 3], True),
                                                 num_parallel_calls=4)

        dataset = tf.data.Dataset.zip((reflectance, background, envmap)).repeat().batch(FLAGS.batch_size).prefetch(
            buffer_size=4)
        iterator = dataset.make_one_shot_iterator()
        input_batch = iterator.get_next()
        refl = tf.map_fn(tf.image.per_image_standardization, input_batch[0])
        bg = tf.map_fn(tf.image.per_image_standardization, input_batch[1])
        gt = tf.map_fn(tf.image.per_image_standardization, input_batch[2])

        prediction = self.inference((refl, bg, gt))
        self.loss_calculation(prediction, gt)
        self.train_op = self.optimize()
        self.summaries = self.summary(bg, refl, gt, prediction)
        return

    """Calculate a prediction RM and intermediary sparse RM"""

    def inference(self, input_batch):
        with tf.device('/gpu:0'):
            return self.encode(input_batch[0], input_batch[1])

    """Calculate the l2 norm loss between the prediction and ground truth"""

    def loss_calculation(self, prediction, gt):
        self.loss = tf.sqrt(tf.reduce_sum(tf.square(gt - prediction)), name="l2_norm")

    """Use Gradient Descent Optimizer to minimize loss"""

    def optimize(self):
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def singlet(self, input):
        encode_1 = encode_decode.encode_layer(input, 64, (3, 3), (2, 2), 2)
        encode_2 = encode_decode.encode_layer(encode_1, 128, (3, 3), (2, 2), 2)
        encode_3 = encode_decode.encode_layer(encode_2, 256, (3, 3), (2, 2), 3)
        encode_4 = encode_decode.encode_layer(encode_3, 512, (3, 3), (2, 2), 3)
        encode_5 = encode_decode.encode_layer(encode_4, 512, (3, 3), (2, 2), 3)
        return [encode_1, encode_2, encode_3, encode_4, encode_5]

    def encode(self, reflectance_map, background):

        rm_encodings = self.singlet(reflectance_map)
        bg_encodings = self.singlet(background)

        fully_encoded = tf.concat([rm_encodings[-1], bg_encodings[-1]], axis=-1)
        # fully_encoded = tf.nn.dropout(fully_encoded, 0.5)
        decode_1 = encode_decode.decode_layer(fully_encoded, 512, (3, 3), (2, 2), 3)
        decode_2 = encode_decode.decode_layer(tf.concat([decode_1, rm_encodings[3]], axis=-1), 512, (3, 3), (2, 2), 3)
        decode_3 = encode_decode.decode_layer(tf.concat([decode_2, rm_encodings[2]], axis=-1), 256, (3, 3), (2, 2), 3)
        decode_4 = encode_decode.decode_layer(tf.concat([decode_3, rm_encodings[1]], axis=-1), 128, (3, 3), (2, 2), 1)
        self.test = (decode_4)
        return encode_decode.encode_layer(decode_4, 3, (3, 3), (1, 1), 1)

    """Create tensorboard summaries of images and loss"""

    def summary(self, bg, reflectance, gt, pred):

        # adjusted_gt = tf.map_fn(tf.image.per_image_standardization,gt)
        # adjusted_pred = tf.map_fn(tf.image.per_image_standardization,pred)
        img_out_summary = tf.summary.image('Generated Envmap', pred, max_outputs=1)
        img_in_summary = tf.summary.image('Ground Truth Envmap', gt, max_outputs=1)
        img_bg_summary = tf.summary.image('Input Background', bg, max_outputs=1)
        img_reflectance_summary = tf.summary.image('Input Reflectance Map', reflectance, max_outputs=1)
        loss_summary = tf.summary.scalar("Loss", self.loss)
        img_summary = tf.summary.merge(
            [img_out_summary, img_in_summary, img_bg_summary, img_reflectance_summary])
        return loss_summary, img_summary

    """Train the model with the settings provided in FLAGS"""

    def train(self, sess=None):

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        if sess is None:
            sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        options, run_metadata = None, None
        if FLAGS.debug:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        tf.train.start_queue_runners(sess=sess)
        train_writer = tf.summary.FileWriter("envmap_synthetic_results/{}".format(time.strftime("%H:%M:%S")),
                                             sess.graph)

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
                    #test = sess.run(self.test)
                    #print(test)
                    train_writer.add_summary(summary, epoch * epoch_size + i)
                    train_writer.add_summary(loss_summary, epoch * epoch_size + i)
                    saver.save(sess, os.path.join("dematerial_graph", 'model'), global_step=epoch)
                    if FLAGS.debug:
                        train_writer.add_run_metadata(run_metadata, "step{}".format(epoch * epoch_size + i),
                                                      global_step=None)
                    train_writer.flush()
                    print(err)
                    if FLAGS.debug:
                        return
        print("finished")
        train_writer.close()
        sess.close()
