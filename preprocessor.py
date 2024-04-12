import sys
import cv2
import numpy as np
import pandas as pd
from pvlib.location import lookup_altitude
from sklearn.preprocessing import OneHotEncoder

class Preprocessor():

    def __init__(self, path:str = "data/GLC24_PA_metadata_train.csv", 
                 is_train:bool = True):
        """Loads in the raw csv data and optimises data types 

        args:
            path        string  :   The path to load the data from
            is_train    boolean :   True if training data false if test data
        """
        self.data = pd.read_csv(path)
        self.is_train = is_train
        if is_train:
            self.data.speciesId = self.data.speciesId.astype("uint16")
        self.data.areaInM2 = self.data.areaInM2.astype("float16")

    def preprocess(self):
        """Runs through the preprocessing steps as follows:
            - Drops unnecessary columns
            - Replaces -inf and nan in areaInM2 with the median value
            - Selects the 500 most common species in the training data
            - One hot encodes the species IDs in training data
            - Gets the elevation values for a given longtitude and latitude
        """
        self._drop_columns()
        self._fix_nans()
        if self.is_train:
            self._reduce_species()
        self._get_elevations()
        if self.is_train:
            self._one_hot_encoding()
        else:
            self._one_hot_encoding(columns=["region"])
        return self.data

    def _reduce_species(self, limit: int = 500, is_species_count: bool = True):
        """ Removes the lease common species based on either a total remaining
        species count or a minimum occurence count

        args:
            limit       int     :   The number of species to keep or the minimum
                                    occurence of a species to be kept
            is_species_count    :   True for total number of species false for 
                       boolean      minimum occurrence
        """
        uniques = self.data["speciesId"].value_counts().to_dict()
        if is_species_count:
            retained_species = list(uniques.keys())[:limit]
        else:
            stop_point = -1
            for value in list(uniques.values).reverse():
                stop_point -= 1
                if value > limit: 
                    break 

            retained_species = list(uniques.keys())[:stop_point]
        self.data = self.data[self.data['speciesId'].isin(retained_species)]
                         
    def _get_elevations(self):
        """ Collects and stores the elevation values for a given latitude 
        longtitude pair
        """
        locations = self.__get_unique_locations()
        locations["elevation"] = locations.apply(
            lambda x: lookup_altitude(x["lat"], x["lon"]),
            axis=1
        )
        self.data["elevation"] = self.data["locationString"].map(
            dict(zip(locations["locationString"], locations["elevation"]))
        )
        self.data.elevation = self.data.elevation.astype("int16")
        self.data.drop(columns=["locationString"], inplace=True)
    
    def _drop_columns(self, columns=["geoUncertaintyInM", "country"]):
        """Drops columns from the dataset

        args:
            columns     list    :   The columns to drop
        """
        self.data = self.data.drop(columns=columns, axis=1)

    def _one_hot_encoding(self, columns=["region", "speciesId"]):
        """One hot encodes the data from specified columns, groups by surveyId
        and drops the columns of which the encoding was based

        args:
            columns     list    :   The columns to encode
        """
        one_hot = OneHotEncoder(dtype='bool')
        transformed = one_hot.fit_transform(self.data[columns])
        if len(columns) >= 2:
            self.data[
                np.concatenate(
                    (one_hot.categories_[0], one_hot.categories_[1]), axis=0)
            ] = transformed.toarray()
        else:
            self.data[one_hot.categories_[0]] = transformed.toarray()
        self.data = self.data.groupby("surveyId").max().copy()
        self._drop_columns(columns=columns)

    def _fix_nans(self):
        """Replaces not a number and +/- Infinity values in the areaInM2 column
        with the median value from the column
        """
        with pd.option_context("use_inf_as_na", True):
            self.data.areaInM2 = self.data.areaInM2.fillna(self.data.areaInM2.median())

    def __get_unique_locations(self):
        """Adds a string to the stored dataframe for each unique 
        latitude/longtitude pair

        return
                        pd.DataFrame    A reduced dataframe containing the lat
                                        and lon values as well as their unique
                                        identifiers with no duplicates
        """
        self.data["locationString"] = self.data.apply(
            lambda x: str(x["lat"]) + ',' + str(x["lon"]), 
            axis=1
        )
        return self.data.drop_duplicates(subset="locationString")[
            ["lat", "lon", "locationString"]
        ]
    

class ImagePreprocessor():

    __image_dims = (128, 128, 4)

    def __init__(self, train_ids: list = None, test_ids: list = None):
        self.train_ids = train_ids
        self.test_ids = test_ids

    def preprocess(self):
        if self.train_ids is not None:
            self.__processing_helper(is_train=True)
        if self.test_ids is not None:
            self.__processing_helper(is_train=False)

        return  (
            self.train_images if self.train_ids is not None else None,
            self.test_images if self.test_ids is not None else None
        )


    def __processing_helper(self, is_train: bool):
        id_list = self.train_ids if is_train else self.test_ids

        if is_train and not hasattr(self, "train_images"):
            self.train_images = np.empty(
                (len(self.train_ids),) + self.__image_dims,
                dtype=np.uint8
            )
        if not is_train and not hasattr(self, "test_images"):
            self.test_images = np.empty(
                (len(self.test_ids),) + self.__image_dims,
                dtype=np.uint8
            )

        for index, id in enumerate(id_list):
            img = self.__get_img_from_id(id, is_train=is_train)
            if is_train: self.train_images[index] = img 
            else: self.test_images[index] = img

    def __get_img_from_id(self, id:int, is_train:bool):
        images = []
        for is_rgb in [True, False]:
            file_path = "data/" \
                    + self.__get_img_path(id, is_train=is_train, is_rgb=is_rgb)
            try:
                images.append(cv2.imread(file_path))
            except (FileNotFoundError):
                text = "The generated file path does not correspond with an " \
                     + "extant image file\n::\t ID:\t" + id + "\n::\t Path\t" \
                     + self.__parse_img_id(id)
                Exception(text)
        output = np.concatenate((images[0], images[1]), axis=-1)
        return output[:, :, :4]
    
    def __get_img_path(self, id:int, is_train:bool, is_rgb:bool):
        id = str(id)
        match len(id):
            case 1:
                ab = 1
                cd = id 
            case 2:
                ab = 1
                cd = id 
            case 3:
                ab = id[-3]
                cd = id[-2:]
            case _:
                ab = id[-4:-2]
                cd = id[-2:]

        t_or_t = "Train" if is_train else "Test"
        r_or_n = "RGB" if is_rgb else "NIR"
        
        path = "PA_" + t_or_t + "_SatellitePatches_" + r_or_n + "/pa_" \
             + t_or_t.lower() + "_patches_" + r_or_n.lower() + '/'
        
        return path + '/'.join([str(cd), str(ab), str(id)]) + ".jpeg"


def main():
    dir_path = "processed_data/"
    # filename = "top_500_species_train.pkl"
    filename = "top_500_species_test.pkl"
    # pp = Preprocessor()
    pp = Preprocessor(path="data/GLC24_PA_metadata_test.csv", is_train=False)
    data = pp.preprocess()
    data.to_pickle(f"{dir_path}{filename}")
    print(f":: saved data of shape {data.shape} to {dir_path}{filename}")

if __name__ == "__main__":
    main()
