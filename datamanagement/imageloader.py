import numpy as np
import cv2

class ImageLoader():

    def _get_image_from_id(self, id: int, rgb: bool, train: bool, 
                                                data_path: str = "data/"):
        """
        """
        file_location = data_path + self.__get_dir(train, rgb) \
                      + self.__parse_img_id(id)
        try:
            image_array = cv2.imread(file_location)
        except (FileNotFoundError):
            message = "The generated file path does not correspond with an " \
                    + "extant image file\n::\t ID:\t" + id + "\n::\t Path\t" \
                    + self.__parse_img_id(id)
            Exception(message)
        return image_array
        
    def __get_dir(self, is_train: bool = True, is_rgb: bool = True):
        t_or_t = "Train" if is_train else "Test"
        r_or_n = "RGB" if is_rgb else "NIR"
        path = "PA_" + t_or_t + "_SatellitePatches_" + r_or_n + "/pa_" \
             + t_or_t.lower() + "_patches_" + r_or_n.lower() + '/'
        return path       

    def __parse_img_id(self, id: int):
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

        paresed_id = '/'.join([str(cd), str(ab), str(id)])
        return paresed_id + ".jpeg"

