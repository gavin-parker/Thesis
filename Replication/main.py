from models import reflectance, dematerial, stereo
import preprocessing_ops
import sys


def main():
    if '--reflectance' in sys.argv:
        refl_model = reflectance.Model()
        refl_model.train()
    if '--dematerial' in sys.argv:
        demat_model = dematerial.Model()
        if '--test' in sys.argv:
            demat_model.test_model()
        else:
            demat_model.train()
    if '--stereo' in sys.argv:
        stereo_model = stereo.Model()
        stereo_model.train()
    if '--experiment' in sys.argv:
        demat_model = dematerial.Model()
        demat_loss = demat_model.test_model(model_dir='dematerial_model_dir')
        stereo_model = stereo.Model()
        stereo_loss = stereo_model.test_model(model_dir='stereo_model_dir')
        print("Dematerial score: {}".format(demat_loss))
        print("Stereo score: {}".format(stereo_loss))


if __name__ == "__main__":
    preprocessing_ops.test_depth()
    #main()
