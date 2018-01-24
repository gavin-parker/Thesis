import os
import numpy as np
import matplotlib
print os.getenv("DISPLAY")
#matplotlib.use('GTK')
import matplotlib.pyplot as plt
from scipy import ndimage


class Dataset:
    def __init__(self, generate=False):
        if generate:
            self.generate_dataset()
        self.load_cahed_dataset()
        return


    def partition(self, test_fraction=0.1):
        np.random.rand()

    def load_cahed_dataset(self):
        dataset = np.load("dataset.npz")
        self.envmaps = dataset['envmaps']
        self.samples = dataset['samples']
        print(self.envmaps.shape)
        print(self.samples.shape)

    def generate_dataset(self):
        envmaps = []
        samples = []
        for sample_set in os.listdir("training"):
            render_path = "training/{}/renders".format(sample_set)
            envmap_path = "training/{}/envmap".format(sample_set)
            envmap_names = np.array(os.listdir(envmap_path))
            envmap_sides = np.array([ndimage.imread(envmap_path + "/{}".format(x)) for x in os.listdir(envmap_path)])
            renders = np.array([ndimage.imread(render_path + "/{}".format(x)) for x in os.listdir(render_path)])
            # 6 sides of 256x256 rgba envmap -> one wide image for tf
            envmap = envmap_sides.reshape(6 * 256, 256, 4)
            envmaps.append(envmap)
            samples.append(renders)
            #print("sample: {}, size: {}".format(sample_set,renders.shape))

        envmaps = np.array(envmaps)
        samples = np.concatenate(np.array(samples))
        np.savez_compressed("dataset.npz", envmaps=envmaps, samples=samples)
