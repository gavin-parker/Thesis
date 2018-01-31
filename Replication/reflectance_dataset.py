import numpy as np
import os
from scipy import ndimage
import tensorflow as tf


class ReflectanceDataset:
    normals = []
    inputs = []
    ground_truths = []
    train_path = "synthetic/synthetic/train"
    sess = []

    def __init__(self, sess):
        self.sess = sess
        self.inputs = tf.train.match_filenames_once("{}/radiance/*.png".format(self.train_path))
        self.normals = tf.train.match_filenames_once("{}/normal/*.png".format(self.train_path))
        self.ground_truths = tf.train.match_filenames_once("{}/lit/*.png".format(self.train_path))
        return

    def get_batch(self, indices):
        input_batch = self.inputs[indices]
        normal_batch = self.normals[indices]
        ground_truths = self.ground_truths[indices]
        return self.load_images(input_batch), \
               self.load_images(normal_batch), \
               self.load_images(ground_truths)

    def load_images(self, names_tensor):
        return tf.map_fn(lambda x: tf.image.decode_png(x), names_tensor)
