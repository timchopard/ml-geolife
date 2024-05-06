import numpy as np
import pandas as pd
import pickle as pk

COLORS = ["red", "green", "blue", "nir", "swir1", "swir2"]

def get_path(is_train, color):
    is_train = "train" if is_train else "test"
    path = f"data/_raw/PA-{is_train}-landsat_time_series/GLC24-PA-{is_train}"
    path += f"-landsat_time_series-{color}.csv"
    return path 

def pickle_landsat():
    for truth in [True, False]:
        image_data = None 
        for c_idx, color in enumerate(COLORS):
            path = get_path(truth, color)
            data = pd.read_csv(path)
            if image_data is None:
                image_data = np.zeros((data.shape[0], 60, 6))
            cols = data.columns.to_list()
            cols.reverse()
            data = data[cols]
            data["from"] = data.apply(
                lambda x: str(int(x.first_valid_index()[:4]) - 15) + "_1",
                axis=1
            )
            data.reset_index()
            for idx, row in data.iterrows():
                for modifier in range(0, 15):
                    for quarter in range(1, 5):
                        image_data[
                            idx, 4 * modifier + quarter - 1, c_idx
                        ] = np.uint8(
                            row[f"{int(row['from'][:4]) + modifier}_{quarter}"]
                        )
        save_path = f"data/processed/full_{'train' if truth else 'test'}"
        save_path += "_landsat.pkl"
        pk.dump(image_data, open(save_path, "wb"))


if __name__ == "__main__":
    pickle_landsat()