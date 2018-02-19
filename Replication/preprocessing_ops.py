import tensorflow as tf
import glob
import imageio
from skimage import color
import numpy as np

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


def preprocess_color(filename, input_shape=[256, 256, 3], double_precision=False, use_lab=False):
    image = tf.image.decode_image(tf.read_file(filename), channels=3, name="decode_color")
    if not double_precision:
        image = tf.cast(format_image(image, input_shape, 128), tf.float16)
    else:
        image = tf.cast(format_image(image, input_shape, 128), tf.float32)
    #if use_lab:
    #    new_image = tf.py_func(rgb_to_lab, [image], tf.float32)
    #    image = tf.reshape(new_image, tf.shape(image))
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
    image = tf.reshape(rgbe_image, [128,128,3])
    image = tf.image.resize_images(image, [64, 64])
    return image


def parse_hdr(filename):
    img = imageio.imread(filename, format='hdr')
    return img


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


def image_stream(path):
    return tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(sorted(glob.glob(path)), dtype=tf.string))


# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
# Shamelessly ripped from pix2pix https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
def rgb_to_lab(rgb):
    lab = color.rgb2lab(rgb).astype(np.float32)
    return lab


def lab_to_rgb(lab):
    return color.lab2rgb(lab).astype(np.float32)
