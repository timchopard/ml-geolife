import os
import pickle as pk
from pipeline2 import GeoLifePreprocessor, GeoLifePCA
from pipeline2.preprocessing import pickle_landsat

class SetupException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__("Please import the raw data into data/_raw/")

if __name__ == "__main__":
    if not os.path.exists("data/_raw"):
        raise SetupException()
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")

    train, test = GeoLifePreprocessor().process_data()
    train, test = GeoLifePCA(train, test).process_data()
    pk.dump(train, open("data/processed/train.pkl", "wb"))
    pk.dump(test, open("data/processed/test.pkl", "wb"))
    train_ls, test_ls = pickle_landsat(train)
    pk.dump(train_ls, open("data/processed/train_ls", "wb"))
    pk.dump(test_ls, open("data/processed/test_ls", "wb"))