import tensorflow as tf
from trainer import preprocessing_ops as preprocessing
from models import layers
# from rendering import renderer as rend
from trainer.params import FLAGS
from trainer import reflectance_ops

""" Convolutional-Deconvolutional model for extracting Environment maps from stereo images."""

"""Defines the graph and provides a template training function"""

regularizer = tf.contrib.layers.l1_regularizer(scale=FLAGS.weight_decay)


class Model:
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 1, 0.95, staircase=False)
    if not FLAGS.app and not FLAGS.test:
        validation_step = tf.placeholder_with_default(False, [])
        train_dataset = preprocessing.get_stereo_dataset(FLAGS.train_dir, FLAGS.batch_size)
        val_dataset = preprocessing.get_stereo_dataset(FLAGS.val_dir, FLAGS.batch_size)
        iter_train_handle = train_dataset.make_one_shot_iterator().string_handle()
        iter_val_handle = val_dataset.make_one_shot_iterator().string_handle()
        handle = tf.placeholder(tf.string, shape=[])
    name = 'stereo'

    def __init__(self):
        with tf.device('/gpu:0'):
            if FLAGS.app or FLAGS.test:
                self.left_image = tf.placeholder(tf.float32, shape=[1, 256, 256, 3])
                self.right_image = tf.placeholder(tf.float32, shape=[1, 256, 256, 3])
                self.norm_image = tf.placeholder(tf.float32, shape=[1, 64, 64, 3])
                self.bg_image = tf.placeholder(tf.float32, shape=[1, 256, 256, 3])
                gt = tf.placeholder(tf.float32, shape=[1, 64, 64, 3])
                pass
            else:
                iterator = tf.data.Iterator.from_string_handle(
                    self.handle, self.train_dataset.output_types, self.train_dataset.output_shapes)
                train_batch = iterator.get_next()
                self.left_image = train_batch[0]
                self.right_image = train_batch[1]
                self.norm_image = train_batch[3]
                self.norm_image = tf.image.resize_images(self.norm_image, (64, 64))
                self.bg_image = train_batch[4]
                gt = train_batch[2]

            if FLAGS.use_bg:
                l_intensity = tf.reduce_sum(self.left_image, axis=3)
                r_intensity = tf.reduce_sum(self.right_image, axis=3)
                l_mask = tf.cast(tf.equal(l_intensity, 0), tf.float32)
                r_mask = tf.cast(tf.equal(r_intensity, 0), tf.float32)
                l_mask = tf.stack([l_mask, l_mask, l_mask], axis=3)
                r_mask = tf.stack([r_mask, r_mask, r_mask], axis=3)
                self.left_image = self.bg_image * l_mask + self.left_image
                self.right_image = self.bg_image * r_mask + self.right_image
            else:
                self.bg_image = tf.map_fn(preprocessing.rgb_to_lab, self.bg_image)
            gt_norm = preprocessing.normalize_hdr(gt)
            left_lab = tf.map_fn(preprocessing.rgb_to_lab, self.left_image)
            right_lab = tf.map_fn(preprocessing.rgb_to_lab, self.right_image)
            gt_lab = tf.map_fn(preprocessing.rgb_to_lab, gt_norm)
            predictions = self.inference((left_lab, right_lab, gt_lab))
            self.diff = tf.abs(tf.reduce_max(gt_lab) - tf.reduce_max(predictions[0]))
            self.loss_calculation(predictions, (gt_lab, self.norm_image))
            self.train_op = self.optimize()
            self.check = [tf.is_nan(predictions), tf.is_nan(self.loss)]

            pred_pretty = tf.map_fn(preprocessing.lab_to_rgb, predictions[0])
            self.converted_prediction = preprocessing.denormalize_hdr(pred_pretty)
            self.summaries = self.summary(self.left_image, self.right_image, gt_lab, predictions, gt, left_lab,
                                          right_lab)
            self.gt = gt
            self.validate()
        return

    """Calculate a prediction RM and intermediary sparse RM"""

    def inference(self, input_batch):
        return self.encode(input_batch[0], input_batch[1])

    """Calculate the l2 norm loss between the prediction and ground truth"""

    def loss_calculation(self, (env_pred, norm_pred), (env_gt, norm_gt)):
        self.env_loss = tf.reduce_sum(tf.abs(env_pred - env_gt))
        self.l2_norm_env = tf.reduce_sum(tf.square(env_pred - env_gt))
        self.norm_loss = tf.reduce_sum(tf.abs(norm_pred - norm_gt))
        self.loss = self.env_loss
        if FLAGS.norms:
            self.loss += self.norm_loss

    def gabriel_loss(self, prediction_log, gt_log):
        n = 1.0 / (3.0 * 64 * 64)
        return n * tf.reduce_sum(tf.square(prediction_log - gt_log))

    """Use Gradient Descent Optimizer to minimize loss"""

    def optimize(self):
        if FLAGS.adam:
            return tf.train.AdamOptimizer(1e-5).minimize(self.loss)
        else:
            return tf.train.MomentumOptimizer(self.learning_rate, 0.95).minimize(self.loss)

    def encode(self, left, right):

        l_encoding, l_encodings_downsampled, _ = layers.siamese_encode_2(left, reuse=False, depth=FLAGS.depth)
        r_encoding, r_encodings_downsampled, r_multiscale = layers.siamese_encode_2(right, reuse=True,
                                                                                    depth=FLAGS.depth)
        if FLAGS.dotprod:
            l_end = tf.nn.l2_normalize(l_encoding, dim=3)
            r_end = tf.nn.l2_normalize(r_encoding, dim=3)
            joint = tf.matmul(l_end, r_end, transpose_b=True)
            joint = tf.concat([l_encoding, r_encoding, joint], axis=-1)
        elif FLAGS.dotprod_pyramid:
            joint = tf.concat([l_encoding, r_encoding], axis=-1)
            for l, r in zip(l_encodings_downsampled, r_encodings_downsampled):
                l_norm = tf.nn.l2_normalize(l, dim=3)
                r_norm = tf.nn.l2_normalize(r, dim=3)
                similarity = tf.matmul(l_norm, r_norm, transpose_b=True)
                joint = tf.concat([joint, similarity], axis=-1)
        else:
            joint = tf.concat([l_encoding, r_encoding], axis=-1)
        if FLAGS.sep_bg:
            bg_encoding = layers.basic_encode(self.bg_image, depth=FLAGS.depth)
            joint = tf.concat([joint, bg_encoding], axis=-1)
        joint = layers.encode_layer(joint, 512, (3, 3), (2, 2), 2, maxpool=False)
        joint_b = layers.encode_layer(joint, 512, (3, 3), (2, 2), 2, maxpool=True)
        joint_c = layers.encode_layer(joint_b, 1024, (3, 3), (2, 2), 1, maxpool=True)

        env_pred = self.decode(joint_c, r_multiscale)
        if FLAGS.norms:
            norm_pred = self.decode(joint_c, r_multiscale)
            return env_pred, norm_pred
        return env_pred, env_pred

    def decode(self, encoded, multiscale):
        if FLAGS.multiscale:
            encoded = tf.concat([encoded, multiscale[-1]], axis=-1)
            decode_1 = layers.decode_layer(encoded, 512, (3, 3), (2, 2), FLAGS.depth)
            decode_1 = tf.concat([decode_1, multiscale[-2]], axis=-1)
            decode_2 = layers.decode_layer(decode_1, 512, (3, 3), (2, 2), FLAGS.depth)
            decode_2 = tf.concat([decode_2, multiscale[-3]], axis=-1)
            decode_3 = layers.decode_layer(decode_2, 256, (3, 3), (2, 2), FLAGS.depth)
        else:
            decode_1 = layers.decode_layer(encoded, 512, (3, 3), (2, 2), FLAGS.depth)
            decode_2 = layers.decode_layer(decode_1, 512, (3, 3), (2, 2), FLAGS.depth)
            decode_3 = layers.decode_layer(decode_2, 256, (3, 3), (2, 2), FLAGS.depth)
        return layers.encode_layer(decode_3, 3, (1, 1), (1, 1), 1, activation=None, norm=False, maxpool=False)

    """Create tensorboard summaries of images and loss"""

    def summary(self, left, right, gt_lab, preds, gt, left_lab, right_lab):
        self.pred_lab = preds[0]

        summaries = [tf.summary.image('Generated_Envmap', preds[0], max_outputs=1),
                     tf.summary.image('Ground_Truth_Envmap', gt_lab, max_outputs=1),
                     tf.summary.image('Original_GT_Envmap', gt, max_outputs=1),
                     tf.summary.image('Converted_generated_Envmap', self.converted_prediction, max_outputs=1),
                     tf.summary.image('Input_Left_Image', left, max_outputs=1),
                     tf.summary.image('CIELAB_Left', left_lab, max_outputs=1),
                     tf.summary.image('Input_Right_Image', right, max_outputs=1),
                     tf.summary.image('CIELAB_Right', right_lab, max_outputs=1),
                     tf.summary.image('GT_norms', self.norm_image, max_outputs=1),
                     tf.summary.image('Background', self.bg_image, max_outputs=1)]
        if FLAGS.norms:
            summaries.append(tf.summary.image('Predicted_norms', preds[1], max_outputs=1))
        scalar_summary = tf.summary.merge([tf.summary.scalar("Loss", self.loss),
                                           tf.summary.scalar("Lighting_Loss", self.env_loss),
                                           tf.summary.scalar("Norm_Loss", self.norm_loss),
                                           tf.summary.scalar("Max_difference", self.diff),
                                           tf.summary.scalar("Learning_Rate", self.learning_rate)])
        img_summary = tf.summary.merge(summaries)
        return img_summary, scalar_summary

    def validate(self):
        validation_loss = self.l2_norm_env
        with tf.variable_scope("validation_mean") as scope:
            self.val_loss, self.val_update = tf.metrics.mean(validation_loss)
            self.validation_summary = tf.summary.merge([tf.summary.scalar("Validation_Loss", self.val_loss)])
            stream_vars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            self.reset_mean = [tf.variables_initializer(stream_vars)]
