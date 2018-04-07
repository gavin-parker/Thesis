from models import reflectance, dematerial, stereo
from params import FLAGS

import harness
import sys


def main():
    model = None
    if '--reflectance' in sys.argv:
        model = reflectance.Model()
    if '--dematerial' in sys.argv:
        model = dematerial.Model()
    if '--stereo' in sys.argv:
        model = stereo.Model()
    name = get_name(model)
    print("Model Name: {}".format(name))
    harness.train(model, name=name)


def get_name(model):
    name_flags = ['learning_rate', 'batch_size', 'max_epochs', 'lab_space']
    name = model.name
    for flag in FLAGS.__flags:
        if flag in name_flags:
            name += "&{}={}".format(flag, FLAGS.__flags[flag])
    return name


if __name__ == "__main__":
    main()
