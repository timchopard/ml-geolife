""" from helper import load_images
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

def load_images(labels:list, is_train:bool = True):
    """Imports four channel png images from either the train or test data set
    and returns them as a numpy array ordered by the id list provided.

    args:
        labels          :   List of the surveyId labels corresponding to the 
                            desired images
        is_train        :   True if images are from the training data, False if
                            from the testing data

    return:
        np.array            An array of int8 values containg the loaded images
    """
    if not os.path.isdir("data-pipeline"):
        raise Exception(":: [ERROR] this is designed to be run from the root " \
                        + "directory, however cannot find \'data-pipeline\'")
    images = np.zeros((len(labels), 128, 128, 4), dtype=np.uint8)
    image_path = os.path.join(
        "data-pipeline/output/2-all_images",
        "train" if is_train else "test"
    )
    for idx, img_id in enumerate(tqdm(labels)):
        path = os.path.join(image_path, f"{img_id}.png")
        try:
            images[idx] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except:
            message = f":: [ERROR] image with ID {img_id} not found at "
            message += f"\'{path}\'"
            raise Exception(message)
    return images
    