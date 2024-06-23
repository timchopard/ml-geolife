import os
import sys
import pickle as pk
from pipeline import GeoLifePreprocessor, GeoLifePCA
from pipeline.preprocessing import pickle_landsat

class SetupException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__("Please import the raw data into data/_raw/")

if __name__ == "__main__":

    species_drop = 100

    args = sys.argv
    if '--species' in args and len(args) > args.index('--species') + 1:
        species = args[args.index('--species') + 1]
        if species.isdigit() and int(species) > 0 and int(species) < 1000:
            species_drop = int(species)
    print(f"Dropping species below {species_drop} occurrences")

    if not os.path.exists("data/_raw"):
        raise SetupException()
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")

    train, test = GeoLifePreprocessor().process_data(species_drop=species_drop)
    train, test = GeoLifePCA(train, test).process_data()
    pk.dump(train, open("data/processed/train.pkl", "wb"))
    pk.dump(test, open("data/processed/test.pkl", "wb"))
    # train_ls, test_ls = pickle_landsat(train)
    pickle_landsat(train)
    # pk.dump(train_ls, open("data/processed/train_ls", "wb"))
    # pk.dump(test_ls, open("data/processed/test_ls", "wb"))