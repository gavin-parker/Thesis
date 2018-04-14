from models import stereo, reflectance, stereo_deeper
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
    if '--stereo2' in sys.argv:
        model = stereo_deeper.Model()
    name = get_name(model)
    print("Model Name: {}".format(name))
    if '--test' in sys.argv:
        harness.collect_results(model)
    else:
        harness.train(model, name=name)


def get_name(model):
    name_flags = ['learning_rate', 'batch_size', 'max_epochs', 'lab_space', 'log_prefix', 'dotprod']
    name = model.name
    for flag in FLAGS.__flags:
        if flag in name_flags:
            name += "&{}={}".format(flag, FLAGS.__flags[flag])
    return name


if __name__ == "__main__":
        main()
