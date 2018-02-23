import tensorflow as tf
import os
import preprocessing_ops as preprocessing
import encode_decode
import time
from params import FLAGS
import render_master
import cv2
import numpy as np

""" Convolutional-Deconvolutional model for extracting reflectance maps from input images with normals.
    Extracts sparse reflectance maps, with the CNN performing data interpolation"""

"""Defines the graph and provides a template training function"""

regularizer = tf.contrib.layers.l1_regularizer(scale=FLAGS.weight_decay)


class Model:
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                               1, 0.95, staircase=False)
    synth_path = 'synthetic'
    train_path = 'train'
    if FLAGS.validate:
        train_path = 'val'
    if FLAGS.real:
        synth_path = 'real'
    train_path = FLAGS.train_dir + "{}/{}".format(synth_path, train_path)
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
                buffer_size=2 * FLAGS.batch_size)
            iterator = dataset.make_one_shot_iterator()
            input_batch = iterator.get_next()
            refl = input_batch[0]
            bg = input_batch[1]
            gt = input_batch[2]
        with tf.device('/gpu:0'):
            gt_norm = preprocessing.normalize_hdr(gt)
            bg_lab, refl_lab, gt_lab = bg, refl, gt
            if FLAGS.use_lab:
                bg_lab = tf.map_fn(preprocessing.rgb_to_lab, bg)
                refl_lab = tf.map_fn(preprocessing.rgb_to_lab, refl)
                gt_lab = tf.map_fn(preprocessing.rgb_to_lab, gt_norm)
                prediction = self.inference((refl_lab, bg_lab, gt_lab))
                self.diff = tf.abs(tf.reduce_max(gt_lab) - tf.reduce_max(prediction))
                self.loss_calculation(prediction, gt_lab)
            else:
                prediction = self.inference((refl, bg, gt_norm))
                self.diff = tf.abs(tf.reduce_max(gt_norm) - tf.reduce_max(prediction))
                self.loss_calculation(prediction, gt_norm)
            self.train_op = self.optimize()
            self.summaries = self.summary(bg, refl, gt_lab, prediction, gt, bg_lab, refl_lab)
            self.gt = gt
            self.gt_lab = gt_lab
            self.converted_gt = tf.map_fn(preprocessing.lab_to_rgb, gt_lab)
            self.converted_gt = tf.map_fn(preprocessing.denormalize_hdr, self.converted_gt)
            # self.gt_lab = gt_lab
        return

    """Calculate a prediction RM and intermediary sparse RM"""

    def inference(self, input_batch):
        return self.encode(input_batch[0], input_batch[1])

    """Calculate the l2 norm loss between the prediction and ground truth"""

    def loss_calculation(self, prediction, gt_lab):
        self.loss = tf.reduce_sum(tf.abs(prediction - gt_lab)) + tf.losses.get_regularization_losses()

    def gabriel_loss(self, prediction_log, gt_log):
        n = 1.0 / (3.0 * 64 * 64)
        return n * tf.reduce_sum(tf.square(prediction_log - gt_log))

    """Use Gradient Descent Optimizer to minimize loss"""

    def optimize(self):
        return tf.train.MomentumOptimizer(self.learning_rate, 0.95).minimize(self.loss)
        #return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)

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
        fully_encoded = tf.nn.dropout(fully_encoded, 0.5)
        fully_encoded = encode_decode.encode_layer(fully_encoded, 1024, (3, 3), (1, 1), 1, maxpool=False)
        decode_1 = encode_decode.decode_layer(fully_encoded, 512, (3, 3), (2, 2), 3)
        decode_2 = encode_decode.decode_layer(tf.concat([decode_1, rm_encodings[3]], axis=-1), 512, (3, 3), (2, 2), 3)
        decode_3 = encode_decode.decode_layer(tf.concat([decode_2, rm_encodings[2]], axis=-1), 256, (3, 3), (2, 2), 3)
        decode_4 = encode_decode.decode_layer(tf.concat([decode_3, rm_encodings[1]], axis=-1), 128, (3, 3), (2, 2), 1)
        return encode_decode.decode_layer(decode_4, 3, (2, 2), (1, 1), 0, activation=None)

    """Create tensorboard summaries of images and loss"""

    def summary(self, bg, reflectance, gt_lab, pred, gt, bg_lab, refl_lab):
        self.pred_lab = pred
        pred_pretty = tf.map_fn(preprocessing.lab_to_rgb, pred)
        self.converted_prediction = preprocessing.denormalize_hdr(pred_pretty)
        summaries = [tf.summary.image('Generated Envmap', pred, max_outputs=1),
                     tf.summary.image('Ground Truth Envmap', gt_lab, max_outputs=1),
                     tf.summary.image('Original GT Envmap', gt, max_outputs=1),
                     tf.summary.image('Converted generated Envmap', self.converted_prediction, max_outputs=1),
                     tf.summary.image('Input Background', bg, max_outputs=1),
                     tf.summary.image('CIELAB Background', bg_lab, max_outputs=1),
                     tf.summary.image('Input Reflectance Map', reflectance, max_outputs=1),
                     tf.summary.image('CIELAB Reflectance Map', refl_lab, max_outputs=1)]

        loss_summary = tf.summary.scalar("Loss", self.loss)
        lr_summary = tf.summary.scalar("Max difference", self.diff)
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
        if FLAGS.fine_tune:
            saver.restore(sess, FLAGS.test_model_dir)
        epoch_size = 56238 / FLAGS.batch_size
        print("beginning training with learning rate: {}".format(FLAGS.learning_rate))
        print("Epoch size: {}".format(epoch_size))
        print("Batch size: {}".format(FLAGS.batch_size))
        # generate batches and run graph
        for epoch in range(0, FLAGS.max_epochs):
            t0 = time.time()
            for i in range(0, epoch_size):
                sess.run([self.loss, self.train_op], options=options,
                         run_metadata=run_metadata)
                if i % 100 == 0:
                    err, (summary, loss_summary, diff_summary) = sess.run(
                        [self.loss, self.summaries], options=options,
                        run_metadata=run_metadata)
                    t1 = time.time()
                    train_writer.add_summary(summary, epoch * epoch_size + i)
                    train_writer.add_summary(loss_summary, epoch * epoch_size + i)
                    train_writer.add_summary(diff_summary, epoch * epoch_size + i)
                    saver.save(sess, os.path.join("dematerial_graph", 'model'))
                    if FLAGS.debug:
                        train_writer.add_run_metadata(run_metadata, "step{}".format(epoch * epoch_size + i),
                                                      global_step=None)
                    train_writer.flush()
                    print("Loss:{}".format(err))
                    print("{} sec per sample".format((t1 - t0) / (100 * FLAGS.batch_size)))
                    if FLAGS.debug:
                        return
        print("finished")
        train_writer.close()
        sess.close()

    def test_model(self, sess=None):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        if sess is None:
            sess = tf.Session(config=config)

        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.test_model_dir)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        test_size = 20
        master = render_master.Master('/home/gavin/blender-2.79-linux-glibc219-x86_64/blender')
        mask = np.sum(cv2.imread('mask.png'), axis=2).astype(np.bool)

        for i in range(0, test_size):
            loss, prediction, gt, pred_lab, gt_lab = sess.run(
                [self.loss, self.converted_prediction, self.converted_gt, self.pred_lab, self.gt_lab])
            print("loss: {}".format(loss))
            if loss < 1:
                preprocessing.write_hdr('prediction.hdr', prediction[0])
                preprocessing.write_hdr('gt.hdr', gt[0])
                master.start_worker('test_elephant.blend', 'gt.hdr', 'gt.png')
                master.start_worker('test_elephant.blend', 'prediction.hdr', 'pred.png')
                gt_elephant = cv2.imread('gt.png')
                pred_elephant = cv2.imread('pred.png')
                background = cv2.imread('bg_gt.png')
                background[mask] = gt_elephant[mask]
                cv2.imwrite('gt_render.png', background)
                background[mask] = pred_elephant[mask]
                cv2.imwrite('pred_render.png', background)
                print("rendered new elephant")
                return

        sess.close()
        return
