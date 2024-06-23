# from pipeline.preprocess import GeoLifePreprocessor
import os 
import numpy as np 
import pandas as pd

class GeoLifePreprocessor():
    """ Collects the raw data from the Metadata and Environmental Rasters into a
    single set of master data frames
    """
    __drop_list = [         # Occurences after group by survey:
        "PANNONIAN",            # 248 occurences 
        "STEPPIC",              # 48 occurences 
        "BLACK SEA",            # 8 occurences
        "Hungary",              # 29 occurences
        "Ireland",              # 12 occurences
        "Monaco",               # 7 occurences
        "The former Yugoslav Republic of Macedonia", # 6 occurences
        "Portugal",             # 6 occurences
        "Norway",               # 5 occurences
        "Andorra",              # 4 occurences
        "Luxembourg",           # 2 occurences
    ]
    __reassign_list = [
        "Bosnia and Herzegovina",
        "Bulgaria",
        "Slovenia",
        "Greece",
        "Serbia",
        "Slovakia",
        "Croatia",
    ]
    _rasters = None 
    _data = None

    def __init__(self, base_path:str = "data/_raw") -> None:
        self._base_path = base_path 
        self._raster_path = os.path.join(
            self._base_path, 
            "EnvironmentalRasters",
            "EnvironmentalRasters"
        )

    def process_data(self, species_drop: int=100) -> None: 
        """ Loads in and processes the data for both test and train datasets
        """
        train = None 
        test = None
        for purpose in ["train", "test"]:
            self._combine_rasters(is_train=(purpose == "train"))
            self._get_metadata(
                is_train=(purpose == "train"), 
                species_drop=species_drop
            )
            self._encode_data(is_train=(purpose == "train"))
            self._data.replace([np.inf, -np.inf], np.nan, inplace=True)
            self._data = self._data.fillna(self._data.median())
            if purpose == "train":
                train = self._data.join(self._rasters)
                print("Train done")
            else:
                test = self._data.join(self._rasters)
                print("Test done")
        return train, test

    def _combine_rasters(self, is_train:bool) -> None:
        """
        """
        self._rasters = None 
        for raster in [
            "Elevation",
            "Human Footprint",
            "LandCover",
            "SoilGrids",
        ]:
            loaded = pd.read_csv(
                os.path.join(
                    self._raster_path,
                    self.__get_filepath(is_train=is_train, dir_name=raster)
                ), index_col="surveyId"
            )
            self._rasters = loaded if self._rasters is None                    \
                                   else self._rasters.join(loaded)
            
        self._rasters = self._rasters.join(pd.read_csv(
            os.path.join(
            self._raster_path,
            "Climate",
            "Monthly",
            self.__get_filepath(
                is_train=is_train, 
                dir_name="bioclimatic monthly").split('/')[1]
            ), index_col="surveyId")
        )

    def _get_metadata(self, is_train:bool, species_drop:int = 100) -> None:
        """
        """
        if not is_train:
            self._data = pd.read_csv(
                os.path.join(self._base_path, "GLC24_PA_metadata_test.csv"),
                index_col="surveyId"
            )

            return 
        
        self._data = pd.read_csv(
            os.path.join(self._base_path, "GLC24_PA_metadata_train.csv"),
            index_col="surveyId"
        )

        drop_list = []
        for species, count in self._data["speciesId"]                           \
                                            .value_counts().to_dict().items():
            if count < species_drop:
                drop_list.append(species)
        self._data = self._data[~self._data["speciesId"].isin(drop_list)]

    def _encode_data(self, is_train) -> None:
        """
        """
        if is_train:
            self._data["speciesId"] = self._data["speciesId"].astype(np.int16)
            species = pd.get_dummies(
                self._data["speciesId"],
                prefix="speciesId",
            ).groupby(self._data.index).max()
            species = species.groupby(species.index).max()
            self._data.drop(columns="speciesId", inplace=True)
        self._data = self._data.groupby(self._data.index).first()

        # Drop or reassign outliers
        self._data = self._data[~self._data["country"].isin(self.__drop_list)]
        self._data = self._data[~self._data["region"].isin(self.__drop_list)]
        self._data = self._data.replace(self.__reassign_list, "temp")

        # One hot encode country and region 
        self._data = self._data.join(pd.get_dummies(self._data["country"]))
        self._data = self._data.join(pd.get_dummies(self._data["region"]))
        self._data.drop(columns=["temp", "country", "region"], inplace=True)

        if not is_train: 
            return
        
        self._data = self._data.join(species)

    def __get_filepath(self, is_train: bool, dir_name: str) -> str:
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