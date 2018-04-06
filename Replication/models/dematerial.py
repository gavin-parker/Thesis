import tensorflow as tf
import os
import preprocessing_ops as preprocessing
import layers
import time
import reflectance_ops
from rendering import renderer as rend
from params import FLAGS

""" Convolutional-Deconvolutional model for extracting reflectance maps from input images with normals.
    Extracts sparse reflectance maps, with the CNN performing data interpolation"""

"""Defines the graph and provides a template training function"""

regularizer = tf.contrib.layers.l1_regularizer(scale=FLAGS.weight_decay)


class Model:
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,1, 0.95, staircase=False)
    synth_path = 'synthetic'
    train_path = 'train'
    if FLAGS.validate:
        train_path = 'val'
    if FLAGS.real:
        synth_path = 'real'
    train_path = FLAGS.train_dir + "{}/{}".format(synth_path, train_path)
    train_dataset = preprocessing.get_dematerial_dataset(FLAGS.train_dir, FLAGS.batch_size)
    val_dataset = preprocessing.get_dematerial_dataset(FLAGS.val_dir, FLAGS.batch_size)
    iter_train_handle = train_dataset.make_one_shot_iterator().string_handle()
    iter_val_handle = val_dataset.make_one_shot_iterator().string_handle()
    handle = tf.placeholder(tf.string, shape=[])

    def __init__(self):
        with tf.device('/cpu:0'):
            iterator = tf.data.Iterator.from_string_handle(
                self.handle, self.train_dataset.output_types, self.train_dataset.output_shapes)
            (rgb_image, rgb_normals, envmap) = iterator.get_next()
            self.sphere = preprocessing.get_norm_sphere(FLAGS.batch_size)
            self.sphere = preprocessing.unit_norm(self.sphere[:, :, :, :3], channels=3)
            self.zero_mask = preprocessing.zero_mask(rgb_normals)
            appearance = rgb_image * self.zero_mask
            rgb_normals = rgb_normals * self.zero_mask
            bg = rgb_image * preprocessing.flip_mask(self.zero_mask)
            bg = tf.cast(bg, tf.float32)
        with tf.device('/gpu:0'):
            normal_sphere_flat = tf.reshape(self.sphere, [FLAGS.batch_size, -1, 3])
            sparse_rm = tf.map_fn(reflectance_ops.online_reflectance,
                                  (appearance, rgb_normals, normal_sphere_flat), dtype=tf.float16,
                                  name="reflectance")
            refl = tf.cast(tf.reshape(sparse_rm, [FLAGS.batch_size, 128, 128, 3]), dtype=tf.float32,
                                name="recast")

            gt_norm = preprocessing.normalize_hdr(envmap)
            bg_lab, refl_lab, gt_lab = bg, refl, gt_norm
            if FLAGS.lab_space:
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
            pred_pretty = tf.map_fn(preprocessing.lab_to_rgb, prediction)
            self.converted_prediction = preprocessing.denormalize_hdr(pred_pretty)
            self.summaries = self.summary(bg, refl, gt_lab, prediction, envmap, bg_lab, refl_lab)
            render_sim, envmap_sim, pred_render, gt_render = tf.py_func(rend.render_summary, [self.converted_prediction[0], envmap[0]], [tf.float32, tf.float32, tf.uint8, tf.uint8])
            render_image_summaries = tf.summary.merge([tf.summary.image('Ground Truth Render', tf.expand_dims(gt_render,0), max_outputs=1),
                                        tf.summary.image('Predicted Render', tf.expand_dims(pred_render,0), max_outputs=1)])
            render_similarities = tf.summary.merge([tf.summary.scalar('Render Similarity', render_sim),
                                        tf.summary.scalar('Envmap Similarity', envmap_sim)])
            self.render_summary = [render_similarities, render_image_summaries]
            self.validate()
        return

    """Calculate a prediction RM and intermediary sparse RM"""

    def inference(self, input_batch):
        return self.encode(input_batch[0], input_batch[1])

    """Calculate the l2 norm loss between the prediction and ground truth"""

    def loss_calculation(self, prediction, gt_lab):
        prediction = tf.image.resize_images(prediction, [64,64])
        self.loss = tf.reduce_sum(tf.abs(prediction - gt_lab))

    def gabriel_loss(self, prediction_log, gt_log):
        n = 1.0 / (3.0 * 64 * 64)
        return n * tf.reduce_sum(tf.square(prediction_log - gt_log))

    """Use Gradient Descent Optimizer to minimize loss"""

    def optimize(self):
        #return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)
        return tf.train.MomentumOptimizer(self.learning_rate, 0.95).minimize(self.loss)

    @staticmethod
    def singlet(input):
        encode_1 = layers.encode_layer(input, 64, (3, 3), (2, 2), 2, regularizer=regularizer)
        encode_2 = layers.encode_layer(encode_1, 128, (3, 3), (2, 2), 2, regularizer=regularizer)
        encode_3 = layers.encode_layer(encode_2, 256, (3, 3), (2, 2), 3, regularizer=regularizer)
        encode_4 = layers.encode_layer(encode_3, 512, (3, 3), (2, 2), 3, regularizer=regularizer)
        encode_5 = layers.encode_layer(encode_4, 512, (3, 3), (2, 2), 3, regularizer=regularizer)
        return [encode_1, encode_2, encode_3, encode_4, encode_5]

    def encode(self, reflectance_map, background):

        rm_encodings = self.singlet(reflectance_map)
        bg_encodings = self.singlet(background)

        fully_encoded = tf.concat([rm_encodings[-1], bg_encodings[-1]], axis=-1)
        #fully_encoded = tf.nn.dropout(fully_encoded, 0.5)
        fully_encoded = layers.encode_layer(fully_encoded, 1024, (3, 3), (1, 1), 1, maxpool=False)
        decode_1 = layers.decode_layer(fully_encoded, 512, (3, 3), (2, 2), 3)
        decode_2 = layers.decode_layer(tf.concat([decode_1, rm_encodings[3]], axis=-1), 512, (3, 3), (2, 2), 3)
        decode_3 = layers.decode_layer(tf.concat([decode_2, rm_encodings[2]], axis=-1), 256, (3, 3), (2, 2), 3)
        decode_4 = layers.decode_layer(tf.concat([decode_3, rm_encodings[1]], axis=-1), 128, (3, 3), (2, 2), 1)
        return layers.encode_layer(decode_4, 3, (1, 1), (1, 1), 1, activation=None, norm=False, maxpool=False)

    """Create tensorboard summaries of images and loss"""

    def summary(self, bg, reflectance, gt_lab, pred, gt, bg_lab, refl_lab):
        self.pred_lab = pred
        summaries = [tf.summary.image('Generated Envmap', pred, max_outputs=1),
                     tf.summary.image('Ground Truth Envmap', gt_lab, max_outputs=1),
                     tf.summary.image('Original GT Envmap', gt, max_outputs=1),
                     tf.summary.image('Converted generated Envmap', self.converted_prediction, max_outputs=1),
                     tf.summary.image('Input Background', bg, max_outputs=1),
                     tf.summary.image('CIELAB Background', bg_lab, max_outputs=1),
                     tf.summary.image('Input Reflectance Map', reflectance, max_outputs=1),
                     tf.summary.image('CIELAB Reflectance Map', refl_lab, max_outputs=1),
                     tf.summary.image('Gauss Sphere', self.sphere, max_outputs=1)]

        scalar_summary = tf.summary.merge([tf.summary.scalar("Loss", self.loss),
                                          tf.summary.scalar("Max difference", self.diff),
                                          tf.summary.scalar("Learning Rate", self.diff)])
        img_summary = tf.summary.merge(summaries)
        return img_summary, scalar_summary

    def validate(self):
        validation_loss = self.loss
        self.val_loss, self.val_update = tf.metrics.mean(validation_loss)
        self.validation_summary = tf.summary.merge([tf.summary.scalar("Validation Loss", self.val_loss)])