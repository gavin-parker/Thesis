from dataset import Dataset
#from model import Model
from reflectance_dataset import ReflectanceDataset
import reflectance_model
import dematerial_model
import tensorflow as tf


def main():
    #refl_model = reflectance_model.Model()
    #refl_model.train()
    demat_model = dematerial_model.Model()
    demat_model.train()

if __name__ == "__main__":
    main()
