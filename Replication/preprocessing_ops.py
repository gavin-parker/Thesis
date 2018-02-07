import tensorflow as tf

class Preprocessor:
    @staticmethod
    def zero_mask(image):
        flat = tf.reshape(image, [-1, 3])
        totals = tf.reduce_sum(flat, axis=-1)
        flat_mask = tf.not_equal(totals, 0)
        mask = tf.stack([flat_mask] * 3, axis=-1)
        mask = tf.to_float(mask)
        return tf.reshape(mask, tf.shape(image))
    @staticmethod
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
    @staticmethod
    def get_norm_sphere(single_sphere, batch_size):
        batch_sphere = tf.stack([single_sphere] * batch_size, axis=0)
        # mask = zero_mask(batch_sphere)
        batch_sphere = tf.map_fn(lambda x: Preprocessor.format_image(x, [1024, 1024, 3], 1024), batch_sphere, dtype=tf.float32)
        # batch_sphere = tf.multiply(batch_sphere, mask)
        batch_sphere = tf.reshape(batch_sphere, [batch_size, 1024, 1024, 3])
        return tf.cast(tf.image.resize_images(batch_sphere, [128, 128]), dtype=tf.float16)
    @staticmethod
    def unit_norm(image, channels=3):
        flat = tf.reshape(image, [-1, channels])
        norm = tf.sqrt(tf.reduce_sum(tf.square(flat), 1, keep_dims=True))
        norm_flat = flat / norm
        norm_flat = tf.where(tf.is_nan(norm_flat), tf.zeros_like(norm_flat), norm_flat)
        return tf.reshape(norm_flat, tf.shape(image))
    @staticmethod
    def preprocess_color(filename):
        image = tf.image.decode_image(tf.read_file(filename), channels=3, name="decode_color")
        return tf.cast(Preprocessor.format_image(image, [256, 256, 3], 128), tf.float16)
    @staticmethod
    def preprocess_orientation(filename):
        image = tf.image.decode_image(tf.read_file(filename), channels=4, name="decode_norm")
        orientation = Preprocessor.unit_norm(
            Preprocessor.format_image(image, [256, 256, 3], 128, mask_alpha=True)[:, :, :3], channels=3)
        return tf.cast(orientation, tf.float16)
    @staticmethod
    def preprocess_gt(filename):
        image = tf.image.decode_image(tf.read_file(filename), channels=3, name="decode_gt")
        return Preprocessor.format_image(image, [256, 256, 3], 32)

