import tensorflow as tf
import cv2
import numpy as np
import cv2
import preprocessing_ops as preprocessing
import math

def graph():
    inputs = []
    outputs = []
    f = tf.read_file('/home/gavin/workspace/test_bg.png')
    bg = tf.image.decode_image(f)
    outputs.append(bg)
    return outputs


def test_reflectance():
    batch_size = 1
    norms = []
    rgbs = []
    spheres = []
    sphere = tf.image.decode_image(tf.read_file('normal_sphere.png'))
    sphere = tf.image.convert_image_dtype(sphere, tf.float32)
    sphere = tf.nn.l2_normalize(sphere, dim=-1)
    for i in range(1,1+batch_size):
        n = tf.read_file('/home/gavin/scene_data/val/norms/{}.png'.format(i))
        rgb = tf.read_file('/home/gavin/scene_data/val/right/{}.png'.format(i))
        norms.append(tf.image.decode_image(n))
        rgbs.append(tf.image.decode_image(rgb))
        spheres.append(sphere)

    n_batch = tf.stack(norms)
    n_batch = tf.image.convert_image_dtype(n_batch, tf.float32)
    n_batch = tf.nn.l2_normalize(n_batch, dim=-1)
    rgb_batch = tf.stack(rgbs)
    rgb_float = tf.image.convert_image_dtype(rgb_batch, tf.float32)

    sphere_batch = tf.stack(spheres)
    flat_spheres = tf.cast(tf.reshape(sphere_batch, [batch_size,-1,3]), tf.float16)
    flat_norms = tf.cast(tf.reshape(sphere_batch, [batch_size,-1,3]), tf.float16)
    intensities = tf.norm(rgb_float, axis=-1)
    #find cosine similarities between norm image and norm sphere
    sims = tf.matmul(flat_spheres, flat_norms, transpose_b=True)
    #if angle diff < 5 degrees, we say its the same surface normal
    reflectance = tf.multiply(tf.cast(tf.greater(sims, math.cos(0.0872665)), dtype=tf.float16), flat_intensity)
    indices = tf.argmax(reflectance, axis=-1, output_type=tf.int32, name="max_reflectance")

    ref_map = tf.gather(rgb_batch, indices)
    sess = tf.Session()

    pretty_refl = tf.image.convert_image_dtype(reflectance, tf.uint8)
    res = sess.run(intensities)
    print(np.shape(res))
    print(np.min(res))
    print(np.max(res))
    print(np.sum(res))
    print(res)
    cv2.imwrite("norms.png",res[0])
    #cv2.waitKey(1000)


def test():
    sess = tf.Session()
    outs = graph()
    res = np.array((540, 960))
    pixels = np.indices(res)
    pixels[1] = (pixels[1] + 0.5) / 540
    pixels[0] = (pixels[0] + 0.5) / 960
    print(pixels)
    adjusted = np.ones((540, 960))
    adjusted[0] = (adjusted[0] + 0.5) / 960
    adjusted[1] = (adjusted[1] + 0.5) / 540
    origin = [0, 0, 0]

    print(adjusted)
    bg = sess.run(outs[0])
    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
    cv2.imshow('result', bg)

    cv2.waitKey(1000)
    print(np.shape(bg))


if __name__ == "__main__":
    test_reflectance()
