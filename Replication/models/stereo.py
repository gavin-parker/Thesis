import tensorflow as tf
import preprocessing_ops as preprocessing
import layers
from rendering import renderer as rend
from params import FLAGS

""" Convolutional-Deconvolutional model for extracting reflectance maps from input images with normals.
    Extracts sparse reflectance maps, with the CNN performing data interpolation"""

"""Defines the graph and provides a template training function"""

regularizer = tf.contrib.layers.l1_regularizer(scale=FLAGS.weight_decay)


class Model:
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 1, 0.95, staircase=False)
    validation_step = tf.placeholder_with_default(False, [])
    train_dataset = preprocessing.get_stereo_dataset(FLAGS.train_dir, FLAGS.batch_size)
    val_dataset = preprocessing.get_stereo_dataset(FLAGS.val_dir, FLAGS.batch_size)
    iter_train_handle = train_dataset.make_one_shot_iterator().string_handle()
    iter_val_handle = val_dataset.make_one_shot_iterator().string_handle()
    handle = tf.placeholder(tf.string, shape=[])

    def __init__(self):
        with tf.device('/cpu:0'):

            iterator = tf.data.Iterator.from_string_handle(
                self.handle, self.train_dataset.output_types, self.train_dataset.output_shapes)
            train_batch = iterator.get_next()
            left_image = train_batch[0]
            right_image = train_batch[1]
            gt = train_batch[2]
        with tf.device('/gpu:0'):
            gt_norm = preprocessing.normalize_hdr(gt)
            left_lab = tf.map_fn(preprocessing.rgb_to_lab, left_image)
            right_lab = tf.map_fn(preprocessing.rgb_to_lab, right_image)
            gt_lab = tf.map_fn(preprocessing.rgb_to_lab, gt_norm)
            prediction = self.inference((left_lab, right_lab, gt_lab))
            self.diff = tf.abs(tf.reduce_max(gt_lab) - tf.reduce_max(prediction))
            self.loss_calculation(prediction, gt_lab)
            self.train_op = self.optimize()
            pred_pretty = tf.map_fn(preprocessing.lab_to_rgb, prediction)
            self.converted_prediction = preprocessing.denormalize_hdr(pred_pretty)
            self.summaries = self.summary(left_image, right_image, gt_lab, prediction, gt, left_lab, right_lab)
            render_sim, envmap_sim, pred_render, gt_render = tf.py_func(rend.render_summary,
                                                                        [self.converted_prediction[0], gt[0]],
                                                                        [tf.float32, tf.float32, tf.uint8, tf.uint8])
            render_image_summaries = tf.summary.merge(
                [tf.summary.image('Ground Truth Render', tf.expand_dims(gt_render, 0), max_outputs=1),
                 tf.summary.image('Predicted Render', tf.expand_dims(pred_render, 0), max_outputs=1)])
            render_similarities = tf.summary.merge([tf.summary.scalar('Render Similarity', render_sim),
                                                    tf.summary.scalar('Envmap Similarity', envmap_sim)])
            self.render_summary = [render_similarities, render_image_summaries]
            self.gt = gt
            self.validate()
        return

    """Calculate a prediction RM and intermediary sparse RM"""

    def inference(self, input_batch):
        return self.encode(input_batch[0], input_batch[1])

    """Calculate the l2 norm loss between the prediction and ground truth"""

    def loss_calculation(self, prediction, gt_lab):
        self.loss = tf.reduce_sum(tf.abs(prediction - gt_lab))

    def gabriel_loss(self, prediction_log, gt_log):
        n = 1.0 / (3.0 * 64 * 64)
        return n * tf.reduce_sum(tf.square(prediction_log - gt_log))

    """Use Gradient Descent Optimizer to minimize loss"""

    def optimize(self):
        # return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)
        return tf.train.MomentumOptimizer(self.learning_rate, 0.95).minimize(self.loss)

    def encode(self, left, right):

        l_encodings = layers.siamese_encode(left, reuse=False)
        r_encodings = layers.siamese_encode(right, reuse=True)

        joint = tf.concat([l_encodings[-1], r_encodings[-1]], axis=-1)
        # fully_encoded = tf.nn.dropout(fully_encoded, 0.5)
        joint_b = layers.encode_layer(joint, 512, (3, 3), (2, 2), 1, maxpool=True)
        joint_c = layers.encode_layer(joint_b, 512, (3, 3), (2, 2), 1, maxpool=True)
        joint_d = layers.encode_layer(joint_c, 1024, (3, 3), (2, 2), 1, maxpool=True)

        decode_1 = layers.decode_layer(joint_d, 512, (3, 3), (2, 2), 3)
        decode_2 = layers.decode_layer(decode_1, 512, (3, 3), (2, 2), 3)
        decode_3 = layers.decode_layer(decode_2, 256, (3, 3), (2, 2), 3)
        decode_4 = layers.decode_layer(decode_3, 128, (3, 3), (2, 2), 1)

        return layers.encode_layer(decode_4, 3, (1, 1), (1, 1), 1, activation=None, norm=False, maxpool=False)

    """Create tensorboard summaries of images and loss"""

    def summary(self, left, right, gt_lab, pred, gt, left_lab, right_lab):
        self.pred_lab = pred
        summaries = [tf.summary.image('Generated Envmap', pred, max_outputs=1),
                     tf.summary.image('Ground Truth Envmap', gt_lab, max_outputs=1),
                     tf.summary.image('Original GT Envmap', gt, max_outputs=1),
                     tf.summary.image('Converted generated Envmap', self.converted_prediction, max_outputs=1),
                     tf.summary.image('Input Left Image', left, max_outputs=1),
                     tf.summary.image('CIELAB Left', left_lab, max_outputs=1),
                     tf.summary.image('Input Right Image', right, max_outputs=1),
                     tf.summary.image('CIELAB Right', right_lab, max_outputs=1)]

        scalar_summary = tf.summary.merge([tf.summary.scalar("Loss", self.loss),
                                           tf.summary.scalar("Max difference", self.diff),
                                           tf.summary.scalar("Learning Rate", self.learning_rate)])
        img_summary = tf.summary.merge(summaries)
        return img_summary, scalar_summary

    def validate(self):
        validation_loss = self.loss
        self.val_loss, self.val_update = tf.metrics.mean(validation_loss)
        self.validation_summary = tf.summary.merge([tf.summary.scalar("Validation Loss", self.val_loss)])