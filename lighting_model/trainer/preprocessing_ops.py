import tensorflow as tf
import cv2
import numpy as np
import math
from StringIO import StringIO
from tensorflow.python.lib.io import file_io

HDR_MIN = 0.0
HDR_MAX = 1.0
EPS = 1e-12

thetas = np.arange(0,128)
rhos = np.arange(0,128)
convmap = np.ones([128,128,2])

#for i in thetas:
#    for j in rhos:
#        i_angle = ((i - 64)/64.0)*180.0
#        j_angle = ((i - 64)/64.0)*180.0
#        x = int(128*math.sin(j_angle)*math.cos(i_angle))
#        y = int(128*math.sin(j_angle)*math.sin(i_angle))
#        convmap[i,j,0] = x
#        convmap[i,j,1] = y
#
#cartMap = tf.convert_to_tensor(convmap)

def get_input_size(file):
    """Horrific opencv bodge for google bucket storage"""
    f = file_io.read_file_to_string(file)
    data = np.asarray(bytearray(f), dtype='uint8')
    image_file = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    return image_file.shape


def zero_mask(image):
    flat = tf.reshape(image, [-1, 3])
    totals = tf.reduce_sum(flat, axis=-1)
    flat_mask = tf.not_equal(totals, 0)
    mask = tf.stack([flat_mask] * 3, axis=-1)
    mask = tf.cast(mask, tf.float16)
    return tf.reshape(mask, tf.shape(image))

def flip_mask(mask):
    bool_mask = tf.cast(mask, tf.bool)
    flipped = tf.logical_not(bool_mask)
    return tf.cast(flipped, tf.float16)


def get_stereo_dataset(dir, batch_size):
    left_files, right_files, env_files, norm_files, bg_files = stereo_stream(dir)
    envmap = env_files[0].map(
        lambda x: preprocess_hdr(x, env_files[1], output_size=64),
        num_parallel_calls=4)
    left = left_files[0].map(
        lambda x: preprocess_color(x, left_files[1], double_precision=True),
        num_parallel_calls=4)
    norms = norm_files[0].map(
        lambda x: preprocess_orientation(x, norm_files[1], double_precision=True),
        num_parallel_calls=4)
    right = right_files[0].map(
        lambda x: preprocess_color(x, right_files[1], double_precision=True),
        num_parallel_calls=4)
    bg = bg_files[0].map(
        lambda x: preprocess_color(x, bg_files[1], double_precision=True),
        num_parallel_calls=4)
    dataset = tf.data.Dataset.zip((left, right, envmap, norms, bg)).repeat().batch(batch_size).prefetch(
        buffer_size=2* batch_size)
    return dataset


def get_dematerial_dataset(dir, batch_size):
    right_files, norm_files, env_files, bg_files = dematerial_stream(dir)
    right = right_files[0].map(
        lambda x: preprocess_color(x, right_files[1], output_size=128, double_precision=False),
        num_parallel_calls=4)
    envmap = env_files[0].map(
        lambda x: preprocess_hdr(x, env_files[1], output_size=64),
        num_parallel_calls=4)
    norms = norm_files[0].map(
        lambda x: preprocess_orientation(x, norm_files[1],output_size=128, double_precision=False),
        num_parallel_calls=4)
    bg = bg_files[0].map(
        lambda x: preprocess_orientation(x, bg_files[1],output_size=128, double_precision=False),
        num_parallel_calls=4)
    dataset = tf.data.Dataset.zip((right, norms, envmap, bg)).shuffle(128).repeat().batch(
        batch_size).prefetch(
        buffer_size=2 * batch_size)
    return dataset


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


def get_norm_sphere(batch_size):
    single_sphere = tf.image.decode_image(tf.read_file("normal_sphere.png"), channels=3)
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


def preprocess_color(filename, input_shape, output_size=256, double_precision=False):
    file = tf.read_file(filename, name="read_image")
    image = tf.image.decode_png(file, channels=3, name="decode_color")
    if not double_precision:
        image = tf.cast(format_image(image, input_shape, output_size), tf.float16)
    else:
        image = tf.cast(format_image(image, input_shape, output_size), tf.float32)
    # if use_lab:
    #    new_image = tf.py_func(rgb_to_lab, [image], tf.float32)
    #    image = tf.reshape(new_image, tf.shape(image))
    return image


def normalize_hdr(image):
    image_log = tf.log(image + 1.0) / 10.0
    return image_log


def denormalize_hdr(image):
    image_delog = tf.exp(image * 10.0) - 1.0
    return image_delog


"""Estimate the surface normals from a depth image"""


def depth_to_normals(image):
    threshold = image.max
    mask = np.less(image, threshold).astype(np.float32)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    dxdz, dydz = np.gradient(image)
    norms = np.ones((image.shape[0], image.shape[1], 3))
    norms[:, :, 0] = -dxdz
    norms[:, :, 1] = -dydz
    len = np.sqrt(np.square(dxdz) + np.square(dydz) + 1.0)
    norms[:, :, 0] /= len
    norms[:, :, 1] /= len
    norms[:, :, 2] /= len
    norms = norms * mask
    pretty_norms = (norms + 1.0) * 0.5
    pretty_norms *= 255
    cv2.imshow('norms', pretty_norms.astype(np.uint8))
    cv2.waitKey(10000)


def preprocess_hdr(filename, input_shape=[32, 32, 3], output_size=64):
    rgbe_image = tf.py_func(parse_hdr, [filename], tf.float32)
    image = tf.reshape(rgbe_image, input_shape)
    image = tf.image.resize_images(image, [output_size, output_size])
    return image


def parse_hdr(filename):
    f = file_io.read_file_to_string(filename)
    data = np.asarray(bytearray(f), dtype='uint8')
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    # img = cv2.resize(image, (256, 256))
    return img.astype(np.float32)


def write_hdr(filename, image):
    cv2.imwrite(filename, image)


def preprocess_orientation(filename, input_shape, output_size=256, double_precision=False):
    image = tf.image.decode_image(tf.read_file(filename), channels=4, name="decode_norm")
    orientation = unit_norm(
        format_image(image, input_shape, output_size, mask_alpha=True)[:, :, :3], channels=3)
    if not double_precision:
        return tf.cast(orientation, tf.float16)
    return tf.cast(orientation, tf.float32)


def preprocess_gt(filename, double_precision=False):
    image = tf.image.decode_image(tf.read_file(filename), channels=3, name="decode_gt")
    return format_image(image, [256, 256, 3], 32)


def image_stream(path):
    files = sorted(file_io.get_matching_files(path))
    train_count = int(len(files) * 0.9)
    training = files[0:train_count]
    validation = files[train_count + 1:-1]
    return tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(training)), tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(validation))


def dematerial_stream(dir):
    right_files = sorted(file_io.get_matching_files("{}/right/*.png".format(dir)))
    norm_files = sorted(file_io.get_matching_files("{}/norms/*.png".format(dir)))
    bg_files = sorted(file_io.get_matching_files("{}/bg/*.png".format(dir)))

    envmap_files = sorted(file_io.get_matching_files("{}/envmaps/*.hdr".format(dir)))
    right_shape = get_input_size(right_files[0])
    norm_shape = get_input_size(norm_files[0])
    envmap_shape = get_input_size(envmap_files[0])
    bg_shape = get_input_size(bg_files[0])

    assert len(right_files) == len(envmap_files)
    assert right_shape == norm_shape
    norms = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(norm_files, dtype=tf.string))
    right = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(right_files, dtype=tf.string))
    envmaps = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(envmap_files, dtype=tf.string))
    bgs = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(bg_files, dtype=tf.string))
    return (right, right_shape), (norms, norm_shape), (envmaps, envmap_shape), (bgs, bg_shape)


def stereo_stream(dir):
    print("loading files...")
    left_files = sorted(file_io.get_matching_files("{}/left/*.png".format(dir)))
    right_files = sorted(file_io.get_matching_files("{}/right/*.png".format(dir)))
    norm_files = sorted(file_io.get_matching_files("{}/norms/*.png".format(dir)))
    envmap_files = sorted(file_io.get_matching_files("{}/envmaps/*.hdr".format(dir)))
    bg_files = sorted(file_io.get_matching_files("{}/bg/*.png".format(dir)))
    print("loaded files")
    left_shape = get_input_size(left_files[0])
    right_shape = get_input_size(right_files[0])
    norms_shape = get_input_size(norm_files[0])
    envmap_shape = get_input_size(envmap_files[0])
    bg_shape = get_input_size(bg_files[0])
    assert len(left_files) == len(right_files) == len(envmap_files) == len(norm_files) == len(bg_files)
    assert left_shape == right_shape
    left = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(left_files, dtype=tf.string))
    right = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(right_files, dtype=tf.string))
    envmaps = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(envmap_files, dtype=tf.string))
    norms = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(norm_files, dtype=tf.string))
    bgs = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(bg_files, dtype=tf.string))
    print("prepared data size: {}".format(len(left_files)))
    return (left, left_shape), (right, right_shape), (envmaps, envmap_shape), (norms, norms_shape), (bgs, bg_shape)


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
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1.0 / 1.088754])

            epsilon = 6.0 / 29.0
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon ** 3.0), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon ** 3.0), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3.0 * epsilon ** 2.0) + 4.0 / 29) * linear_mask + (
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
    l, a, b = tf.split(lab, 3, axis=2)
    l = tf.clip_by_value(l, 0,100)
    a =  tf.clip_by_value(a, -86.185,98.254)
    b =  tf.clip_by_value(b, -107.863,94.482)
    lab = tf.concat([l,a,b], axis=2)
    with tf.name_scope("lab_to_rgb"):
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1.0 / 116.0, 1.0 / 116.0, 1.0 / 116.0],  # l
                [1.0 / 500.0, 0.0, 0.0],  # a
                [0.0, 0.0, -1.0 / 200.0],  # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6.0 / 29.0
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3.0 * epsilon ** 2 * (fxfyfz_pixels - 4.0 / 29.0)) * linear_mask + (
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
                    (rgb_pixels ** (1.0 / 2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))
