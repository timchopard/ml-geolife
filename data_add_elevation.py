import pandas as pd

from datamanagement.dataloader import DataLoader

metadata_files = ["GLC24_P0_metadata_train", "GLC24_PA_metadata_train", "GLC24_PA_metadata_test"]

for file in metadata_files:
    data = DataLoader("data/" + file + ".csv", None)
    data._get_elevations()
    data.train.to_pickle("processed_data/" + file + "_elevation.pkl")