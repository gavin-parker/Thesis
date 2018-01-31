from dataset import Dataset
#from model import Model
from reflectance_dataset import ReflectanceDataset
from reflectance_model import Model
import tensorflow as tf


def main():
    sess = tf.Session()
    refl_data = ReflectanceDataset(sess)
    refl_model = Model(refl_data, sess)
    refl_model.train()
    #a, b, c = refl_data.get_batch(1)
    sess.close()


if __name__ == "__main__":
    main()
