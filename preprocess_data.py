import pickle as pk
from pipeline2 import GeoLifePreprocessor, GeoLifePCA
from pipeline2.preprocessing import pickle_landsat

if __name__ == "__main__":
    train, test = GeoLifePreprocessor().process_data()
    train, test = GeoLifePCA(train, test).process_data()
    pk.dump(train, open("data/processed/train.pkl", "wb"))
    pk.dump(test, open("data/processed/test.pkl", "wb"))
    train_ls, test_ls = pickle_landsat()
    pk.dump(train_ls, open("data/processed/train_ls", "wb"))
    pk.dump(test_ls, open("data/processed/test_ls", "wb"))