from models import  stereo_deeper, normals
import trainer.params
import harness
import sys

def main():
    model = None
    #if '--reflectance' in sys.argv:
    #    model = reflectance.Model()
    #if '--dematerial' in sys.argv:
    #    model = dematerial.Model()
    #if '--bob' in sys.argv:
    #    model = bob.Model()
    if '--normals' in sys.argv:
        model = normals.Model()
    #if '--stereo' in sys.argv:
    #    model = stereo.Model()
    if '--stereo2' in sys.argv:
        model = stereo_deeper.Model()
    name = get_name(model)
    print("Model Name: {}".format(name))
    if '--test' in sys.argv:
        harness.collect_results(model)
    else:
        harness.train(model, name=name)


def get_name(model):
    name_flags = ['learning_rate', 'batch_size', 'max_epochs', 'lab_space', 'log_prefix', 'dotprod', 'dotprod-pyramid', 'multiscale']
    name = model.name
    for flag in trainer.params.FLAGS.__flags:
        if flag in name_flags:
            name += "&{}={}".format(flag, trainer.params.FLAGS.__flags[flag])
    return name


if __name__ == "__main__":
    main()
