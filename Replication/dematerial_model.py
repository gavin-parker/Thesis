import tensorflow as tf
import os
import preprocessing_ops as preprocessing
import encode_decode
import time
from params import FLAGS
import glob
import random
""" Convolutional-Deconvolutional model for extracting reflectance maps from input images with normals.
    Extracts sparse reflectance maps, with the CNN performing data interpolation"""

"""Defines the graph and provides a template training function"""

regularizer = tf.contrib.layers.l1_regularizer(scale=FLAGS.weight_decay)

class Model:
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                               1, 0.95, staircase=False)
    train_path = "/mnt/black/MultiNatIllum/data/multiple_materials_single_object/singlets/synthetic/train"
    bg_files = preprocessing.image_stream("{}/background/*.png".format(train_path))
    envmap_files = preprocessing.image_stream("{}/envmap_latlong/*.hdr".format(train_path))
    reflectance_files = preprocessing.image_stream("{}/reflectanceMap_latlong/*.png".format(train_path))

    def __init__(self):
        with tf.device('/cpu:0'):
            background = self.bg_files.map(
                lambda x: preprocessing.preprocess_color(x, input_shape=[128, 128, 3], double_precision=True),
                num_parallel_calls=4)
            envmap = self.envmap_files.map(
                lambda x: preprocessing.preprocess_hdr(x, 64, double_precision=True, use_lab=False),
                num_parallel_calls=4)
            reflectance = self.reflectance_files.map(
                lambda x: preprocessing.preprocess_color(x, [128, 128, 3], double_precision=True),
                num_parallel_calls=4)

            dataset = tf.data.Dataset.zip((reflectance, background, envmap)).repeat().batch(FLAGS.batch_size).prefetch(
                buffer_size=2*FLAGS.batch_size)
            iterator = dataset.make_one_shot_iterator()
            input_batch = iterator.get_next()
            refl = input_batch[0]
            bg = input_batch[1]
            gt = input_batch[2]


        gt_norm = preprocessing.normalize_hdr(gt)
        bg_lab = tf.map_fn(preprocessing.rgb_to_lab, bg)
        refl_lab = tf.map_fn(preprocessing.rgb_to_lab, refl)
        gt_lab = tf.map_fn(preprocessing.rgb_to_lab, gt_norm)

        self.test = refl_lab
        prediction = self.inference((refl_lab, bg_lab, gt_lab))
        self.loss_calculation(prediction, gt_lab)
        self.train_op = self.optimize()
        self.summaries = self.summary(bg, refl, gt_lab, prediction, gt, bg_lab, refl_lab)
        return

    """Calculate a prediction RM and intermediary sparse RM"""

    def inference(self, input_batch):
        with tf.device('/gpu:0'):
            return self.encode(input_batch[0], input_batch[1])

    """Calculate the l2 norm loss between the prediction and ground truth"""

    def loss_calculation(self, prediction, gt):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.01
        self.reg_loss = reg_constant * sum(reg_losses)
        #self.reg_loss = tf.losses.get_regularization_loss()
        self.loss = tf.reduce_mean(tf.abs(gt -prediction))

    def gabriel_loss(self, prediction_log, gt_log):
        n = 1.0 / (3.0*64*64)
        return n*tf.reduce_sum(tf.square(prediction_log - gt_log))


    """Use Gradient Descent Optimizer to minimize loss"""

    def optimize(self):
        #return tf.train.MomentumOptimizer(self.learning_rate, 0.95).minimize(self.loss)
        return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)

    @staticmethod
    def singlet(input):
        encode_1 = encode_decode.encode_layer(input, 64, (3, 3), (2, 2), 2, regularizer=regularizer)
        encode_2 = encode_decode.encode_layer(encode_1, 128, (3, 3), (2, 2), 2, regularizer=regularizer)
        encode_3 = encode_decode.encode_layer(encode_2, 256, (3, 3), (2, 2), 3, regularizer=regularizer)
        encode_4 = encode_decode.encode_layer(encode_3, 512, (3, 3), (2, 2), 3, regularizer=regularizer)
        encode_5 = encode_decode.encode_layer(encode_4, 512, (3, 3), (2, 2), 3, regularizer=regularizer)
        return [encode_1, encode_2, encode_3, encode_4, encode_5]

    def encode(self, reflectance_map, background):

        rm_encodings = self.singlet(reflectance_map)
        bg_encodings = self.singlet(background)

        fully_encoded = tf.concat([rm_encodings[-1], bg_encodings[-1]], axis=-1)
        fully_encoded = encode_decode.encode_layer(fully_encoded, 1024, (3,3), (1,1), 1, maxpool=False)
        decode_1 = encode_decode.decode_layer(fully_encoded, 512, (3, 3), (2, 2), 3, regularizer=regularizer)
        decode_2 = encode_decode.decode_layer(tf.concat([decode_1, rm_encodings[3]], axis=-1), 512, (3, 3), (2, 2), 3,
                                              regularizer=regularizer)
        decode_3 = encode_decode.decode_layer(tf.concat([decode_2, rm_encodings[2]], axis=-1), 256, (3, 3), (2, 2), 3,
                                              regularizer=regularizer)
        decode_4 = encode_decode.decode_layer(tf.concat([decode_3, rm_encodings[1]], axis=-1), 128, (3, 3), (2, 2), 1,
                                              regularizer=regularizer)
        return encode_decode.decode_layer(decode_4, 3, (2, 2), (1, 1), 0,
                                              regularizer=regularizer, activation=None)

    """Create tensorboard summaries of images and loss"""

    def summary(self, bg, reflectance, gt_lab, pred, gt, bg_lab, refl_lab):
        pred_pretty = tf.map_fn(preprocessing.lab_to_rgb, pred)
        pred_pretty = preprocessing.denormalize_hdr(pred_pretty)
        summaries = [tf.summary.image('Generated Envmap', pred, max_outputs=1),
                     tf.summary.image('Ground Truth Envmap', gt_lab, max_outputs=1),
                     tf.summary.image('Original GT Envmap', gt, max_outputs=1),
                     tf.summary.image('Converted generated Envmap', pred_pretty, max_outputs=1),
                     tf.summary.image('Input Background', bg, max_outputs=1),
                     tf.summary.image('CIELAB Background', bg_lab, max_outputs=1),
                     tf.summary.image('Input Reflectance Map', reflectance, max_outputs=1),
                     tf.summary.image('CIELAB Reflectance Map', refl_lab, max_outputs=1)]

        loss_summary = tf.summary.scalar("Loss", self.loss)
        lr_summary = tf.summary.scalar("Learning Rate", self.learning_rate)
        img_summary = tf.summary.merge(summaries)
        return loss_summary, img_summary, lr_summary

    """Train the model with the settings provided in FLAGS"""

    def train(self, sess=None):

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        if sess is None:
            sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        options, run_metadata = None, None
        if FLAGS.debug:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        tf.train.start_queue_runners(sess=sess)
        train_writer = tf.summary.FileWriter("{}/{}".format(FLAGS.log_dir, time.strftime("%H:%M:%S")),
                                             sess.graph)

        saver = tf.train.Saver()
        epoch_size = 56238 / FLAGS.batch_size
        print("beginning training with learning rate: {}".format(FLAGS.learning_rate))
        print("Epoch size: {}".format(epoch_size))
        print("Batch size: {}".format(FLAGS.batch_size))
        # generate batches and run graph
        for epoch in range(0, FLAGS.max_epochs):
            real_max = 0
            for i in range(0, epoch_size):
                sess.run([self.loss, self.train_op], options=options,
                                   run_metadata=run_metadata)
                if i % 100 == 0:
                    err, (summary, loss_summary, lr_summary), reg_loss = sess.run(
                        [self.loss, self.summaries, self.reg_loss], options=options,
                        run_metadata=run_metadata)
                    train_writer.add_summary(summary, epoch * epoch_size + i)
                    train_writer.add_summary(loss_summary, epoch * epoch_size + i)
                    train_writer.add_summary(lr_summary, epoch * epoch_size + i)
                    saver.save(sess, os.path.join("dematerial_graph", 'model'), global_step=epoch)
                    if FLAGS.debug:
                        train_writer.add_run_metadata(run_metadata, "step{}".format(epoch * epoch_size + i),
                                                      global_step=None)
                    train_writer.flush()
                    print("Loss:{}, Reg Loss: {}".format(err, reg_loss))
                    if FLAGS.debug:
                        return
            print(real_max)
        print("finished")
        train_writer.close()
        sess.close()
