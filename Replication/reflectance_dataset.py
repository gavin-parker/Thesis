import numpy as np
import os
from scipy import ndimage
import tensorflow as tf


class ReflectanceDataset:
    normals = []
    inputs = []
    ground_truths = []

    def __init__(self):
        self.inputs = tf.train.match_filenames_once("{}/radiance/*.png".format(self.train_path))
        self.normals = tf.train.match_filenames_once("{}/normal/*.png".format(self.train_path))
        self.ground_truths = tf.train.match_filenames_once("{}/lit/*.png".format(self.train_path))
        return


