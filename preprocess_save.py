import pickle as pk
from pipeline2 import GeoLifePreprocessor, GeoLifePCA

if __name__ == "__main__":
    train, test = GeoLifePreprocessor().process_data()
    train, test = GeoLifePCA(train, test).process_data()
    pk.dump(train, open("data/processed/train.pkl", "wb"))
    pk.dump(test, open("data/processed/test.pkl", "wb"))