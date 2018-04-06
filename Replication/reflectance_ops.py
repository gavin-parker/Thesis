import tensorflow as tf
import math
import numpy as np


def online_reflectance((color_image, normal_image, normal_sphere_flat)):
    normal_flat = tf.reshape(normal_image, [-1, 3], name="flatten_normals")
    color_flat = tf.reshape(color_image, [-1, 3], name="flatten_rgb")
    intensity_flat = tf.sqrt(tf.reduce_sum(tf.square(color_flat), axis=-1))
    indices = unique_2d(normal_flat, intensity_flat, normal_sphere_flat)

    flat_reflmap = tf.gather(color_flat, indices)

    reflmap = tf.reshape(flat_reflmap, tf.shape(color_image))
    return reflmap


# Computes 128x128 matrix to represent dot products of pixels
def unique_2d(flat_input, flat_intensity, sphere):
    diffs = tf.matmul(sphere, flat_input, transpose_b=True, name="normal_product")
    #discretized = tf.cast(tf.greater(diffs, math.cos(0.0872665)), dtype=tf.float16)
    if tf.__version__ == '1.5.0':
        reflectance = tf.boolean_mask(flat_intensity, tf.greater(diffs, math.cos(0.0872665)), axis=1)
    else:
        reflectance = tf.multiply(tf.cast(tf.greater(diffs, math.cos(0.0872665)), dtype=tf.float16), flat_intensity)
        indices = tf.argmax(reflectance, axis=1, output_type=tf.int32, name="max_reflectance")
    return indices
