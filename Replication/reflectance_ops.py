import tensorflow as tf
import math
import numpy as np


class Reflectance:
    @staticmethod
    # Assuming uint8
    def online_reflectance((color_image, normal_image, normal_sphere)):
        normal_flat = tf.reshape(normal_image, [-1, 3])
        normal_sphere_flat = tf.reshape(normal_sphere, [-1, 3])
        color_flat = tf.reshape(color_image, [-1, 3])
        intensity_flat = tf.sqrt(tf.reduce_sum(tf.square(color_flat), axis=-1))
        indices = unique_2d(normal_flat, intensity_flat, normal_sphere_flat)

        flat_reflmap = tf.gather(color_flat, indices)

        reflmap = tf.reshape(flat_reflmap, tf.shape(color_image))
        return reflmap


# Given a row of pixel intensity product (dot product of RGB), returns those within cos(5)
def near_mask(dot_row):
    mask = tf.cast(tf.greater(dot_row, math.cos(0.0872665)), dtype=tf.float32)
    return mask


# Returns the index of the max intensity for a given orientation
def max_idx(dot_row, flat_intensity):
    mask = near_mask(dot_row)
    #mask = tf.stack([mask,mask,mask],axis=-1)
    reflectance = tf.multiply(mask, flat_intensity)

    idx = tf.argmax(reflectance, axis=0, output_type=tf.int64)
    return idx


# Computes 128x128 matrix to represent dot products of pixels
def unique_2d(flat_input, flat_intensity, sphere):
    diffs = tf.matmul(sphere, flat_input, transpose_b=True)
    #diffs = tf.transpose(diffs)
    indices = tf.map_fn(lambda x: max_idx(x, flat_intensity), diffs, dtype=tf.int64)
    return indices
