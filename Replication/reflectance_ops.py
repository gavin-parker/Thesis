import tensorflow as tf
import math
import functools
import numpy as np
import numpy.ma as ma


def ref_func((color_image, normal_image)):
    res = tf.py_func(offline_reflectance, [color_image, normal_image], [tf.float32])
    res = tf.reshape(res, [128, 128, 3])
    return res


def zs(x):
    return np.zeros(x.shape)


def batch_compute_reflectance(color_images, normal_images):
    return tf.expand_dims(ref_func((color_images[0], normal_images[0])), axis=0)


# Given a row of pixel intensity product (dot product of RGB), returns those within cos(5)
def near_mask(dot_row):
    mask = tf.cast(tf.greater(dot_row, math.cos(0.0872665)), dtype=tf.float32)
    return tf.multiply(mask, dot_row)

# Returns the index of the max intensity for a given orientation
def max_idx(dot_row, flat_intensity):
    mask = near_mask(dot_row)
    reflectance = tf.multiply(mask, flat_intensity)
    idx = tf.argmax(reflectance, axis=0, output_type=tf.int64)
    return idx


# Computes 128x128 matrix to represent dot products of pixels
def unique_2d(flat_input, flat_intensity):
    diffs = tf.matmul(flat_input, flat_input, transpose_b=True)
    indices = tf.map_fn(lambda x: max_idx(x, flat_intensity), diffs, dtype=tf.int64)
    return indices


# Assuming uint8
def online_reflectance((color_image, normal_image)):
    normal_flat = tf.reshape(normal_image, [-1, 3])
    color_flat = tf.reshape(color_image, [-1, 3])
    intensity_flat = tf.reduce_sum(color_flat, axis=-1)
    indices = unique_2d(normal_flat, intensity_flat)

    flat_reflmap = tf.gather(color_flat, indices)

    reflmap = tf.reshape(flat_reflmap, [128, 128, 3])
    return reflmap


def offline_reflectance(color_image, normal_image):
    reflmap = np.zeros((128 * 128, 3), dtype=np.float32)
    normal_flat = np.around(np.floor_divide(np.reshape(normal_image, [-1, 3]), 4), decimals=0)
    color_flat = np.reshape(color_image, [-1, 3])
    intensity_flat = np.sum(color_flat, axis=1)
    uniques, indices = np.unique(normal_flat, return_index=True, axis=0)
    for u, i in zip(uniques, indices):
        # print("norms: {}, pixel: {}".format(normal_flat.shape, u.shape))
        mask = np.all(normal_flat == u, axis=tuple(range(-u.ndim, 0))).astype(float)
        masked = np.multiply(intensity_flat, mask)
        idx = np.argmax(masked)
        val = color_flat[idx]
        reflmap[i] = val
        # mask = np.where(eq_op)
        # print(u)
        # print(mask[1])
    return tf.per_image_standardization(np.reshape(reflmap, [128, 128, 3]).astype(np.float32))
