""" from pipeline.preprocess import DataCollector
"""

import os
import numpy as np
import pandas as pd

class DataCollector():
    """ Collects the raw data and combines it into a set of single master 
    dataframes
    """

    def __init__(self, base_path: str = None, raster_path: str = None,
                 drop_list: list = None, reassign_list: list = None) -> None:
        """ Set default values if no custom values passed in
        """
        self._base_path = "data/_raw" if base_path is None else base_path
        self._raster_path = "EnvironmentalRasters/EnvironmentalRasters" if     \
                            raster_path is None else raster_path
        self._drop_list = [         # Occurences after group by survey:
            "PANNONIAN",            # 248 occurences 
            "STEPPIC",              # 48 occurences 
            "BLACK SEA",            # 8 occurences
            "Hungary",              # 29 occurences
            "Ireland",              # 12 occurences
            "Monaco",               # 7 occurences
            "The former Yugoslav Republic of Macedonia", # 6
            "Portugal",             # 6 occurences
            "Norway",               # 5 occurences
            "Andorra",              # 4 occurences
            "Luxembourg",           # 2 occurences
        ] if drop_list is None else drop_list
        self._reassign_list = [
            "Bosnia and Herzegovina",
            "Bulgaria",
            "Slovenia",
            "Greece",
            "Serbia",
            "Slovakia",
            "Croatia",
        ] if reassign_list is None else reassign_list
        self.rasters = None
        self.data = None

    def process_and_save_data(self) -> None:
        """ Loads in and processes the data for both test and train datasets 
        """
        for is_train in [True, False]:
            purpose = "train" if is_train else "test"
            filename = purpose + "_full.pkl"
            filepath = os.path.join("data", "processed", filename)
            print(f":: [{purpose.upper()}] Loading in Environmental Rasters")
            self.combine_rasters(is_train=is_train)
            print(f":: [{purpose.upper()}] Loading in core data")
            self.get_core_data(is_train=is_train)
            print(f":: [{purpose.upper()}] One hot encoding data")
            self.encode_data(is_train=is_train)
            print(f":: [{purpose.upper()}] Saving data to \'{filepath}\'")
            self.data.join(self.rasters).to_pickle(filepath)


    def combine_rasters(self, is_train: bool) -> None:
        """ Loads in and combines all of the Environmental Raster data, storing
        it in a member variable.

        args:
            is_train            :   True if using train data False for test data
        """
        self.rasters = None
        for raster in [
            "Elevation", 
            "Human Footprint", 
            "LandCover", 
            "SoilGrids"
        ]:
            loaded = pd.read_csv(
                os.path.join(
                    self._base_path,
                    self._raster_path,
                    self._get_filepath(is_train=is_train, dir_name=raster)
                ), index_col="surveyId"
            )
            self.rasters = loaded if self.rasters is None                      \
                                  else self.rasters.join(loaded)

        self.rasters = self.rasters.join(pd.read_csv(
            os.path.join(
                self._base_path,
                self._raster_path,
                "Climate",
                "Monthly",
                self._get_filepath(
                    is_train=is_train,
                    dir_name="bioclimatic monthly").split('/')[1]
            ), index_col="surveyId")
        )

    def get_core_data(self, is_train: bool,
                      species_drop_threshold: int = 100) -> None:
        """ Loads in the main data dropping species with occurrences below a 
        threshold for the train data.

        args:
            is_train            :   True if using train data False for test data
            species_drop_threshold : The frequency of occurrence below which
                                     a species should be dropped

        """
        if not is_train:
            self.data = pd.read_csv(
                os.path.join(self._base_path, "GLC24_PA_metadata_test.csv"),
                index_col="surveyId"
            )
            return

        self.data = pd.read_csv(
            os.path.join(self._base_path, "GLC24_PA_metadata_train.csv"),
            index_col="surveyId"
        )
        drop_list = []
        for species, count in self.data["speciesId"]                           \
                                            .value_counts().to_dict().items():
            if count < species_drop_threshold:
                drop_list.append(species)
        self.data = self.data[~self.data["speciesId"].isin(drop_list)]

    def encode_data(self, is_train) -> None:
        """ One hot encodes the species, country and region data. Removes or 
        reassigns rare occurrences.

        args:
            is_train            :   True if using train data False for test data
        """
        if is_train:
            # One hot encode species
            self.data["speciesId"] = self.data["speciesId"].astype(np.int16)
            species = pd.get_dummies(
                self.data["speciesId"],
                prefix="speciesId",
            ).groupby(self.data.index).max()
            species = species.groupby(species.index).max()
            self.data.drop(columns="speciesId", inplace=True)
        self.data = self.data.groupby(self.data.index).first()

        # Drop or reassign outliers in the data
        self.data = self.data[~self.data["country"].isin(self._drop_list)]
        self.data = self.data[~self.data["region"].isin(self._drop_list)]
        self.data = self.data.replace(self._reassign_list, "temp")

        # One hot encode country and region
        self.data = self.data.join(pd.get_dummies(self.data["country"]))
        self.data = self.data.join(pd.get_dummies(self.data["region"]))
        self.data.drop(
            columns=["temp", "country", "region"],
            inplace=True
        )

        if not is_train:
            return

        # Rejoin data if train
        self.data = self.data.join(species)

    def _get_filepath(self, is_train: bool, dir_name: str):
        """ Helper function. Generates a filepath, predominantly for use in the 
        Environmental Raster data.

        args:
            is_train            :   True if using train data False for test data
            dir_name            :   The name to base the filepath on

        """
        path = dir_name
        filename = f"GLC24-PA-{'train' if is_train else 'test'}-"
        filename += dir_name.lower().replace(' ', '_')
        filename += '.csv'
        return os.path.join(path, filename)
