import numpy as np
import os
import tensorflow as tf
import OpenEXR, Imath
import pickle

def getChannels(file):
    r = file.channel("R", Imath.PixelType(Imath.PixelType.FLOAT))
    g = file.channel("G", Imath.PixelType(Imath.PixelType.FLOAT))
    b = file.channel("B", Imath.PixelType(Imath.PixelType.FLOAT))
    header = file.header()['dataWindow']
    size = (header.max.y - header.min.y + 1, header.max.x - header.min.x + 1)
    r = np.fromstring(r, dtype=np.float32).reshape(size)
    g = np.fromstring(g, dtype=np.float32).reshape(size)
    b = np.fromstring(b, dtype=np.float32).reshape(size)
    return np.stack((r, g, b), axis=2)


class Dataset:
    environments = {}
    data_size = 0
    datapoints = []
    training = []
    test = []

    def __init__(self, location, test_split=0.1, generate=False):
        if generate:
            self.generate(location)
        else:
            envfile = open('envs.pk1', 'rb')
            datafile = open('data.pk1', 'rb')
            self.environments = pickle.load(envfile)
            self.datapoints = pickle.load(datafile)
            envfile.close()
            datafile.close()
        self.partition(test_split)

    def generate(self, location):
        for environment in os.listdir(location):
            file = OpenEXR.InputFile("{}/{}/{}.exr".format(location, environment, environment))
            envmap = getChannels(file)
            self.environments[environment] = envmap
            for object in os.listdir("{}/{}".format(location, environment)):
                sess = tf.Session()
                x_image = tf.placeholder(tf.float32, [1,None, None, 3])
                x_norm = tf.placeholder(tf.float32, [1,None, None, 3])
                if not object.endswith(".exr"):
                    path = "{}/{}/{}".format(location, environment, object)
                    img_file = OpenEXR.InputFile("{}/{}_small.exr".format(path, object))
                    image = np.expand_dims(getChannels(img_file), axis=0)
                    normal = np.expand_dims(np.load("{}/{}_n_small.npy".format(path, object)), axis=0)

                    resized_img = tf.map_fn(lambda x: tf.image.resize_image_with_crop_or_pad(x, 128, 128), x_image)
                    resized_norm = tf.map_fn(lambda x: tf.image.resize_image_with_crop_or_pad(x, 128, 128), x_norm)
                    image, normal = sess.run([resized_img, resized_norm], feed_dict={x_image: image, x_norm: normal})
                    self.datapoints.append((image[0], normal[0], environment))
                sess.close()
        envfile = open('envs.pk1', 'wb')
        datafile = open('data.pk1', 'wb')
        pickle.dump(self.environments, envfile, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.datapoints, datafile, pickle.HIGHEST_PROTOCOL)

    def partition(self, test_split):
        test_count = int(test_split* len(self.datapoints))
        print(test_count)
        np.random.shuffle(self.datapoints)
        self.training, self.test = self.datapoints[test_count:], self.datapoints[:test_count]

    def generate_batches(self):
        np.random.shuffle(self.training)
        for point in self.training:
            yield np.expand_dims(point[0], axis=0), np.expand_dims(point[1], axis=0), np.expand_dims(self.environments[point[2]], axis=0)
