from dataset import Dataset
#from model import Model
from reflectance_dataset import ReflectanceDataset
from reflectance_model import Model
import tensorflow as tf


def main():
    refl_model = Model()
    refl_model.train()

if __name__ == "__main__":
    main()
