import pandas as pd 
import numpy as np

from datamanagement.csvloader import CSVLoader
from datamanagement.imageloader import ImageLoader

class DataLoader(CSVLoader, ImageLoader):
    
    def __init__(self, train, test):
        super().__init__(train, test)


    def preprocess_csv(self, is_train: bool = True):
        data = self.train if is_train else self.test
        if data is None:
            Exception(f":: [ERROR] The {'training' if is_train else 'test'} data was not correctly loaded")

        print(f":: [MAIN] Generating elevation column for {'training' if is_train else 'test'} data.")
        # self._get_elevations()        # TODO temporarily commented out

        data_unique_id = data.drop_duplicates(subset="surveyId")
        print(f":: [DEBUG] Data shape: {data_unique_id.shape}")

        color_type = "rgb"

        data_unique_id[color_type] = data_unique_id.apply(
            lambda x: self._get_image_from_id( 
                id=x["surveyId"], rgb=True, train=True
            ), axis=1)
        data_unique_id.drop(columns=[
            "lon", "lat", "geoUncertaintyInM", "areaInM2", 
            "region", "country", "speciesId"
        ],axis=1)

        np.savez(f"images_as_arrays_{color_type}", data_unique_id.to_numpy())
        
        
    def images_to_numpy(self, is_train: bool = True, is_rgb: bool = True, 
                        path: str = "", compress: bool = True) -> str:
        """Converts images to numpy arrays and saves them as files based on 
        color space (RGB or NIR) and data purpose (training or test)

        args:
            is_train    boolean :   True for processing training data false for
                                    test data
            is_rgv      boolean :   True for processing RGB images, false for 
                                    NIR images
            path        string  :   The directory path to save in
            compress    bool    :   True to compress the images in the .npz file
                                    false to save uncompressed

        return:
                        string  :   The filepath to the saved .npz
        """
        data = self.train if is_train else self.test
        if data is None:
            Exception(f":: [ERROR] The {'training' if is_train else 'test'} data was not correctly loaded")

        data_unique_id = data.drop_duplicates(subset="surveyId")

        color_type = "rgb" if is_rgb else "nir"
        data_type = "train" if is_train else "test"

        data_unique_id[color_type] = data_unique_id.apply(
            lambda x: self._get_image_from_id( 
                id=x["surveyId"], rgb=is_rgb, train=is_train
            ), axis=1)

        to_drop = [ ii for ii in list(data_unique_id.columns) if ii not in ["surveyId", color_type] ]
        
        data_unique_id.drop(columns=to_drop, axis=1)

        save_location = f"{path}{data_type}_images_as_arrays_{color_type}"
        if compress:
            np.savez_compressed(save_location, data_unique_id.to_numpy())
        else:
            np.savez(save_location, data_unique_id.to_numpy())

        return save_location