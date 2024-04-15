# Data Processor
#
# Desired output
# - Full train dataset - for training model for Kaggle submission
# - Full test dataset - for Kaggle submission
# - Internal train dataset - 80% of full train dataset for internal banchmarking
# - Internal test dataset - 20% of full train dataset for internal benchmarking
# - All tabular data in one file

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

print("Processing data\n")

# Load data
print("Load data")
train_df = pd.read_csv("data/GLC24_PA_metadata_train.csv")
test_df = pd.read_csv("data/GLC24_PA_metadata_test.csv")

# Drop: 'Andorra', 'Hungary', 'Ireland', 'Latvia', 'Luxembourg', 'Monaco', 'Norway',
# 'Portugal', 'Romania', 'Serbia', 'The former Yugoslav Republic of Macedonia' they
# are not represented in the test set
print("Drop countries not in test data")
drop_countries = {
    "Andorra",
    "Hungary",
    "Ireland",
    "Latvia",
    "Luxembourg",
    "Monaco",
    "Norway",
    "Portugal",
    "Romania",
    "Serbia",
    "The former Yugoslav Republic of Macedonia",
}
train_df = train_df[~train_df.country.isin(drop_countries)]

print("Drop regions not in test data")
drop_regions = {
    "BLACK SEA",
    "PANNONIAN",
}
test_df = test_df[~test_df.region.isin(drop_regions)]

# set speciesId as int
print("Set speciesId as int")
train_df["speciesId"] = train_df["speciesId"].astype(int)

# Resulting dataframes
print("Resulting dataframes")
print(f" - Train data: {train_df.shape}")
print(f" - Test data: {test_df.shape}")

# Combine all environmental data
print("Combine environmental data")
files_to_combine = [
    "data/EnvironmentalRasters/EnvironmentalRasters/Climate/Average 1981-2010/GLC24-PA-{}-bioclimatic.csv",
    "data/EnvironmentalRasters/EnvironmentalRasters/Climate/Monthly/GLC24-PA-{}-bioclimatic_monthly.csv",
    "data/EnvironmentalRasters/EnvironmentalRasters/Elevation/GLC24-PA-{}-elevation.csv",
    "data/EnvironmentalRasters/EnvironmentalRasters/Human Footprint/GLC24-PA-{}-human_footprint.csv",
    "data/EnvironmentalRasters/EnvironmentalRasters/LandCover/GLC24-PA-{}-landcover.csv",
    "data/EnvironmentalRasters/EnvironmentalRasters/SoilGrids/GLC24-PA-{}-soilgrids.csv",
]

for file in files_to_combine:
    print(" - Processing train data:", file.format("train"))
    train_df = pd.merge(train_df, pd.read_csv(file.format("train")), on="surveyId")

for file in files_to_combine:
    print(" - Processing test data:", file.format("test"))
    test_df = pd.merge(test_df, pd.read_csv(file.format("test")), on="surveyId")

# Resulting dataframes
print("Resulting dataframes")
print(f" - Train data: {train_df.shape}")
print(f" - Test data: {test_df.shape}")

# Handle missing data
print("Handle missing data")


# -inf, inf replaced by NaN
print(" - Replace -inf, inf with NaN")
train_df = train_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.replace([np.inf, -np.inf], np.nan)

# For the train data NaN in geoUncertaintyInM, and areaInM2 replaced with the country median values
print(" - Process NaN values")
for column in ["areaInM2", "geoUncertaintyInM"]:
    print(f"   - Processing train data: {column}")
    for country in tqdm(train_df.country.unique()):
        train_df.loc[train_df.country == country, column] = train_df.query(
            f"country == '{country}'"
        )[column].fillna(train_df.query(f"country == '{country}'")[column].median())

# For the test data NaN in geoUncertaintyInM, and areaInM2 replaced with the training data country
# median values (note this is data leakage, but I believe it is tolerable)
for column in ["areaInM2", "geoUncertaintyInM"]:
    print(f"   - Processing test data: {column}")
    for country in test_df.country.unique():
        test_df.loc[test_df.country == country, column] = test_df.query(
            f"country == '{country}'"
        )[column].fillna(train_df.query(f"country == '{country}'")[column].median())

# Resulting dataframes
print(" - Resulting dataframes")
print(f"   - Train data: {train_df.shape}")
print(f"   - Test data: {test_df.shape}")

for column in list(
    train_df.isna().sum()[train_df.isna().sum() > 0].sort_values(ascending=False).keys()
):
    print(f" - Processing train data: {column}")
    for country in tqdm(train_df.country.unique()):
        train_df.loc[train_df.country == country, column] = train_df.query(
            f"country == '{country}'"
        )[column].fillna(train_df.query(f"country == '{country}'")[column].median())

for column in list(
    train_df.isna().sum()[train_df.isna().sum() > 0].sort_values(ascending=False).keys()
):
    print(f" - Processing train data: {column}")
    for country in tqdm(train_df.country.unique()):
        train_df.loc[train_df.country == country, column] = train_df.query(
            f"country == '{country}'"
        )[column].fillna(train_df.query(f"country == '{country}'")[column].median())

for column in list(
    test_df.isna().sum()[test_df.isna().sum() > 0].sort_values(ascending=False).keys()
):
    print(f" - Processing train data: {column}")
    for country in tqdm(test_df.country.unique()):
        test_df.loc[test_df.country == country, column] = test_df.query(
            f"country == '{country}'"
        )[column].fillna(train_df.query(f"country == '{country}'")[column].median())
# Resulting dataframes
print("Resulting dataframes")
print(f" - Train data: {train_df.shape}")
print(f" - Test data: {test_df.shape}")

# for training data country, region, and speciesId one-hot encoded
print("One-hot encode country, region, and speciesId in training data")
for column in ["country", "region", "speciesId"]:
    print(f" - Processing: {column}")
    train_df = pd.concat(
        [train_df, pd.get_dummies(train_df[column], prefix=column)],
        axis=1,
    )
    train_df = train_df.drop(columns=[column])

# for test data country and region one-hot encoded
print("One-hot encode country and region in test data")
for column in ["country", "region"]:
    print(f" - Processing: {column}")
    test_df = pd.concat(
        [test_df, pd.get_dummies(test_df[column], prefix=column)],
        axis=1,
    )
    test_df = test_df.drop(columns=[column])


# Grouped by surveyId, use max
print("Group by surveyId")
train_df = train_df.groupby("surveyId", as_index=False).max()

# Resulting dataframes
print("Resulting dataframes")
print(f" - Train data: {train_df.shape}")
print(f" - Test data: {test_df.shape}")

# Create directories
print("Create directories")
Path("processed_data/full/train_images").mkdir(parents=True, exist_ok=True)
Path("processed_data/full/test_images").mkdir(parents=True, exist_ok=True)

# Save data
print("Saving data tabular data")
train_df.to_pickle("processed_data/full/train.pkl")
test_df.to_pickle("processed_data/full/test.pkl")


# process images
print("Processing images")
print(" - Convert images to 4 channels")
dfs = [(train_df, "train"), (test_df, "test")]
for df, set_type in dfs:
    print(f"   - Processing {set_type} images")
    for survey_id in tqdm(df.surveyId):
        cd = str(survey_id)[-4:-2]
        ab = str(survey_id)[-2:]
        rgb = Image.open(
            f"data/PA_{set_type.capitalize()}_SatellitePatches_RGB/pa_{set_type}_patches_rgb/{ab}/{cd}/{survey_id}.jpeg"
        )
        nir = Image.open(
            f"data/PA_{set_type.capitalize()}_SatellitePatches_NIR/pa_{set_type}_patches_nir/{ab}/{cd}/{survey_id}.jpeg"
        ).convert("L")
        rgb.putalpha(nir)
        rgb.save(f"processed_data/full/{set_type}_images/{survey_id}.png")
