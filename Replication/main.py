import reflectance_model
import dematerial_model
import tensorflow as tf
import sys



def main():
    if '--reflectance' in sys.argv:
        refl_model = reflectance_model.Model()
        refl_model.train()
    if '--dematerial' in sys.argv:
        demat_model = dematerial_model.Model()
        demat_model.train()

if __name__ == "__main__":
    main()
