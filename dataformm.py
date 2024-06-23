from torch.utils.data import Dataset
import torch

import pickle as pk
import os
from PIL import Image

# import torchvision.transforms as transforms


class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.subset = "test"
        self.pca_data = pk.load(open("data/processed/test.pkl", "rb"))
        self.landsat_data = pk.load(open("data/processed/test_ls.pkl", "rb"))
        self.transform = transform

    def __len__(self):
        return self.pca_data.shape[0]

    def __getitem__(self, idx):
        survey_id = self.pca_data.index[idx]
        # image_data = self.process_patch(survey_id)
        sample = [
            self.pca_data.iloc[idx].tolist(), 
            # image_data, 
            self.landsat_data[idx]
        ]
        return sample, survey_id

    def process_patch(self, survey_id):
        rgb_path = f"data/_raw/PA_{self.subset.title()}_SatellitePatches_RGB/pa_{self.subset}_patches_rgb"
        nir_path = f"data/_raw/PA_{self.subset.title()}_SatellitePatches_NIR/pa_{self.subset}_patches_nir"
        for d in (str(survey_id)[-2:], str(survey_id)[-4:-2]):
            rgb_path = os.path.join(rgb_path, d)
            nir_path = os.path.join(nir_path, d)
        rgb_path = os.path.join(rgb_path, f"{survey_id}.jpeg")
        nir_path = os.path.join(nir_path, f"{survey_id}.jpeg")
        rgb_image = Image.open(rgb_path).convert("RGB")
        rgb_image = self.transform(rgb_image)
        rgb_image = rgb_image.unsqueeze(0)
        nir_image = Image.open(nir_path).convert("L")
        nir_image = self.transform(nir_image)
        nir_image = nir_image.unsqueeze(0)
        image_data = torch.cat([rgb_image, nir_image], dim=1)
        image_data = torch.squeeze(image_data)
        return image_data


class TrainDataset(TestDataset):
    def __init__(self, transform=None):
        self.subset = "train"
        self.pca_data = pk.load(open("data/processed/train.pkl", "rb"))
        self.landsat_data = pk.load(open("data/processed/train_ls.pkl", "rb"))
        self.transform = transform
        label_cols = [col for col in self.pca_data.columns if "speciesId_" in col]
        self.labels = self.pca_data[label_cols]
        self.pca_data = self.pca_data.drop(columns=label_cols)

    def __len__(self):
        return self.pca_data.shape[0]

    def __getitem__(self, idx):
        survey_id = self.pca_data.index[idx]
        # image_data = self.process_patch(survey_id)

        sample = [
            torch.tensor(self.pca_data.iloc[idx].tolist(), dtype=torch.float32),
            # torch.tensor(image_data, dtype=torch.float32),
            torch.tensor(self.landsat_data[idx], dtype=torch.float32),
        ]
        return (
            survey_id,
            sample,
            torch.tensor(self.labels.iloc[idx].tolist(), dtype=torch.float32),
        )
