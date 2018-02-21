import tensorflow as tf
import glob
import cv2
import numpy as np

HDR_MIN = 0
HDR_MAX = 1
EPS = 1e-12


def zero_mask(image):
    flat = tf.reshape(image, [-1, 3])
    totals = tf.reduce_sum(flat, axis=-1)
    flat_mask = tf.not_equal(totals, 0)
    mask = tf.stack([flat_mask] * 3, axis=-1)
    mask = tf.to_float(mask)
    return tf.reshape(mask, tf.shape(image))


def format_image(image, in_shape, out_size, mask_alpha=False, normalize=False):
    alpha = []
    if mask_alpha:
        alpha = tf.clip_by_value(tf.to_float(image[:, :, 3]), 0, 1)
    image = image[:, :, :3]
    image = tf.reshape(image, in_shape)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    if normalize:
        image = tf.map_fn(tf.image.per_image_standardization, image)
    if mask_alpha:
        alpha_mask = tf.stack([alpha, alpha, alpha], axis=-1)
        image = tf.multiply(image, alpha_mask)
    image = tf.reshape(image, in_shape)
    image = tf.image.resize_images(image, [out_size, out_size])

    return image


def get_norm_sphere(single_sphere, batch_size):
    batch_sphere = tf.stack([single_sphere] * batch_size, axis=0)
    # mask = zero_mask(batch_sphere)
    batch_sphere = tf.map_fn(lambda x: format_image(x, [1024, 1024, 3], 1024), batch_sphere,
                             dtype=tf.float32)
    # batch_sphere = tf.multiply(batch_sphere, mask)
    batch_sphere = tf.reshape(batch_sphere, [batch_size, 1024, 1024, 3])
    return tf.cast(tf.image.resize_images(batch_sphere, [128, 128]), dtype=tf.float16)


def unit_norm(image, channels=3):
    flat = tf.reshape(image, [-1, channels])
    norm = tf.sqrt(tf.reduce_sum(tf.square(flat), 1, keep_dims=True))
    norm_flat = flat / norm
    norm_flat = tf.where(tf.is_nan(norm_flat), tf.zeros_like(norm_flat), norm_flat)
    return tf.reshape(norm_flat, tf.shape(image))


def preprocess_color(filename, input_shape=[256, 256, 3], double_precision=False):
    image = tf.image.decode_image(tf.read_file(filename), channels=3, name="decode_color")
    if not double_precision:
        image = tf.cast(format_image(image, input_shape, 128), tf.float16)
    else:
        image = tf.cast(format_image(image, input_shape, 128), tf.float32)
    # if use_lab:
    #    new_image = tf.py_func(rgb_to_lab, [image], tf.float32)
    #    image = tf.reshape(new_image, tf.shape(image))
    return image


def normalize_hdr(image):
    image_log = tf.log(image + 1.0 + EPS)
    image_norm = (image_log - HDR_MIN) / HDR_MAX
    return image_norm


def denormalize_hdr(image):
    image = (image * HDR_MAX) + HDR_MIN
    image = tf.exp(image - 1.0 - EPS)
    return image


"""
Tensorflow implementation of http://www.graphics.cornell.edu/%7Ebjw/rgbe/rgbe.c for RGBE to float decoding

rgbe2float(float *red, float *green, float *blue, unsigned char rgbe[4])
{
  float f;

  if (rgbe[3]) {   /*nonzero pixel*/
    f = ldexp(1.0,rgbe[3]-(int)(128+8));
    *red = rgbe[0] * f;
    *green = rgbe[1] * f;
    *blue = rgbe[2] * f;
  }
  else
    *red = *green = *blue = 0.0;
}

"""


def preprocess_hdr(filename, size=64, double_precision=True, use_lab=False):
    rgbe_image = tf.py_func(parse_hdr, [filename], tf.float32)
    image = tf.reshape(rgbe_image, [128, 128, 3])
    image = tf.image.resize_images(image, [64, 64])
    return image


def parse_hdr(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    return img.astype(np.float32)


def preprocess_orientation(filename, double_precision=False):
    image = tf.image.decode_image(tf.read_file(filename), channels=4, name="decode_norm")
    orientation = unit_norm(
        format_image(image, [256, 256, 3], 128, mask_alpha=True)[:, :, :3], channels=3)
    if not double_precision:
        return tf.cast(orientation, tf.float16)
    return tf.cast(orientation, tf.float32)


def preprocess_gt(filename, double_precision=False):
    image = tf.image.decode_image(tf.read_file(filename), channels=3, name="decode_gt")
    return format_image(image, [256, 256, 3], 32)


def image_stream(path, order=None):
    if order is not None:
        return tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(sorted(glob.glob(path)), dtype=tf.string))
    else:
        return tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(sorted(glob.glob(path)), dtype=tf.string))


# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
# Shamelessly ripped from pix2pix https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (
                        ((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])

            epsilon = 6.0 / 29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon ** 3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon ** 3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3.0 * epsilon ** 2) + 4.0 / 29) * linear_mask + (
                        xyz_normalized_pixels ** (1.0 / 3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

    return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
                [1 / 500.0, 0.0, 0.0],  # a
                [0.0, 0.0, -1 / 200.0],  # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6.0 / 29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3.0 * epsilon ** 2 * (fxfyfz_pixels - 4.0 / 29)) * linear_mask + (
                        fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [3.2404542, -0.9692660, 0.0556434],  # x
                [-1.5371385, 1.8760108, -0.2040259],  # y
                [-0.4985314, 0.0415560, 1.0572252],  # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (
                        (rgb_pixels ** (1 / 2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))
