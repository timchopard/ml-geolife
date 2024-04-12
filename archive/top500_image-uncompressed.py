
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

top500_df = pd.read_pickle("../processed_data/partially_processed/top500_df_elevation_onehot_grouped.pkl")

images = []

for survey_id in tqdm(top500_df.surveyId):
    survey_id = str(survey_id)
    cd = survey_id[-2:]
    ab = survey_id[-4:-2]

    ir_image_path = f'../data/PA_Train_SatellitePatches_NIR/pa_train_patches_nir/{cd}/{ab}/{survey_id}.jpeg'
    pil_ir_img = Image.open(ir_image_path)

    rgb_image_path = f'../data/PA_Train_SatellitePatches_RGB/pa_train_patches_rgb/{cd}/{ab}/{survey_id}.jpeg'
    pil_rgb_img = Image.open(rgb_image_path)

    _, b, g = pil_rgb_img.split()

    Image.merge('RGB', (pil_ir_img, g, b)).save(f'../processed_data/partially_processed/uncompressed/{survey_id}_igb.jpeg')

