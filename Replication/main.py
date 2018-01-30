from dataset import Dataset
from model import Model
def main():
    data = Dataset("/home/gavin/workspace/drexel_natgeom", 0.1, generate=False)
    model = Model()
    model.train(data)

if __name__ == "__main__":
    main()