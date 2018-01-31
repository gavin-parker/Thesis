import tensorflow as tf
import random
import numpy as np
from generator import Generator
from dataset import Dataset

def main():
    dataset = Dataset(generate=True)
    gen = Generator()
    gen.train(dataset)
    return


if __name__ == "__main__":
    main()
