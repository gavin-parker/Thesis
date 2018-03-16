from models import reflectance_model, dematerial_model, stereo
import sys

def main():
    if '--reflectance' in sys.argv:
        refl_model = reflectance_model.Model()
        refl_model.train()
    if '--dematerial' in sys.argv:
        demat_model = dematerial_model.Model()
        if '--test' in sys.argv:
            demat_model.test_model()
        else:
            demat_model.train()
    if '--stereo' in sys.argv:
        stereo_model = stereo.Model()
        stereo_model.train()

if __name__ == "__main__":
    main()
