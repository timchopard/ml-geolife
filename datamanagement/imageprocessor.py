import pandas as pd
import numpy as np
import cv2
import os

class ImageProcessor():

    image_dims = (128, 128, 6)

    def __init__(self, train_ids: list = None, test_ids: list = None, 
                 save_path: str = "processed_images/", 
                 train_filename: str = "train_rgbnir", 
                 test_filename: str = "test_rgbnir"
                 ) -> None:
        self.train_ids = train_ids
        self.test_ids = test_ids 
        self.save_path = save_path
        self.train_filename = train_filename
        self.test_filename = test_filename


    def __save_images(self, to_save, filename: str):
        try: 
            os.mkdir(self.save_path)
        except FileExistsError:
            pass 

        np.savez_compressed(self.save_path + filename, to_save)


    def process_list(self, use_train: bool = True, use_test: bool = False, 
                     save: bool = True):
        if use_test and self.test_ids is None:
            raise Exception(":: [ERROR] No testing data ids in ImageProcessor")
        
        if use_train:
            if self.train_ids is None:
                raise Exception(
                    ":: [ERROR] No training data ids in ImageProcessor")
            
        if use_train: 
            self.__process_list_helper(is_train=True)
            if save:
                self.__save_images(self.train_images, self.train_filename)

        if use_test: 
            self.__process_list_helper(is_train=False)
            if save:
                self.__save_images(self.test_images, self.test_filename)
        
        if not save:
            output = (None if not use_train else self.train_images,
                      None if not use_test else self.test_images)
            return output


    def __process_list_helper(self, is_train: bool):
        id_list = self.train_ids if is_train else self.test_ids
        
        if is_train and not hasattr(self, 'train_images'):
            self.train_images = np.empty((len(self.train_ids),) + self.image_dims, dtype=np.uint8)
        if not is_train and not hasattr(self, 'test_images'):
            self.test_images = np.empty((len(self.test_ids),) + self.image_dims, dtype=np.uint8)

        for index, id in enumerate(id_list):
            img = self.__get_img_from_id(id, is_train=is_train)
            if is_train: self.train_images[index] = img  
            else: self.test_images[index] = img  


    def __get_img_from_id(self, id: int, is_train: bool = True, 
                          combine: bool = True, data_path: str = "data/"):
        """Loads the RGB and NIR images relating to a specified ID number

        args:
            id          integer :   The ID code relating to the images to be 
                                    loaded and returned as arrays
            is_train    boolean :   True if the ID relates to training data, 
                                    false for testing data
            combine     boolean :   True to combine the R, G and B channels with 
                                    the N, I and R channels, false to return
                                    a list containing the two images 
            data_path   string  :   The directory in which the data is located

        return: 
            If combine is true:
                numpy array representing the combined images
            If combine is false:
                a list containing two numpy arrays, one for RGB and one for NIR

        """
        images = []
        for is_rgb in [True, False]:

            # Build file path from constituent parts
            file_path = data_path + self.__get_path(
                id, is_train=is_train, is_rgb=is_rgb)
            
            # Attempt to load in the image
            try:
                images.append(cv2.imread(file_path))
            except (FileNotFoundError):
                text = "The generated file path does not correspond with an " \
                     + "extant image file\n::\t ID:\t" + id + "\n::\t Path\t" \
                     + self.__parse_img_id(id)
                Exception(text)
        
        if combine:
            return np.concatenate((images[0], images[1]), axis=-1)
        else:
            return images


    def __get_path(self, id: int, is_train: bool = True, is_rgb: bool = True
                   ) -> str:
        """Builds the full directory path from the id code, colorspace and 
        data function

        args:
            id          integer :   The unique id code corresponding to the 
                                    RGB/NIR image pair
            is_train    boolean :   True if the data is for training, false 
                                    if the data is for testing
            is_rgb      boolean :   True if the data is in RGB colorspace, false
                                    if it is in NIR

        return: 
                        string  :   The directory path of the file
        """
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
    

