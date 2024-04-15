# Data Processor
#
# Desired output
# - Full train dataset - for training model for Kaggle submission
# - Full test dataset - for Kaggle submission
# - Internal train dataset - 80% of full train dataset for internal banchmarking
# - Internal test dataset - 20% of full train dataset for internal benchmarking
# - All tabular data in one file

import pandas as pd
import shutil
from tqdm import tqdm
from pathlib import Path


print("Processing data\n")

# Load data
print("Load data")
train_df = pd.read_pickle("processed_data/top500/train.pkl")

internal_train_df = train_df.sample(frac=0.8, random_state=113)
internal_test_df = train_df.drop(internal_train_df.index)

print(f"{internal_train_df.shape=}")
print(f"{internal_test_df.shape=}")

print("Export internal training data")
print(" - Make directory")
Path("processed_data/internal/train_images").mkdir(parents=True, exist_ok=True)

print(" - Export tabular data")
internal_train_df.to_pickle("processed_data/internal/train.pkl")

internal_train_df.head()

print(" - Copying over images")
for survey_id in tqdm(internal_train_df.surveyId):
    shutil.copy(
        f"processed_data/top500/train_images/{survey_id}.png",
        f"processed_data/internal/train_images/{survey_id}.png",
    )

print("Make test data")
print(" - Make directory")
Path("processed_data/internal/test_images").mkdir(parents=True, exist_ok=True)

print("Split data")
internal_test_X_df = internal_test_df.loc[
    :, ~internal_test_df.columns.str.startswith("speciesId_")
]
species_columns = ["surveyId"] + list(
    internal_test_df.columns[internal_test_df.columns.str.startswith("speciesId_")]
)
internal_test_y_df = internal_test_df.loc[:, species_columns]

print("Save data")
print(" - test data")
internal_test_X_df.to_pickle("processed_data/internal/test_X.pkl")

print(" - Copying over images")
for survey_id in tqdm(internal_test_df.surveyId):
    shutil.copy(
        f"processed_data/top500/train_images/{survey_id}.png",
        f"processed_data/internal/test_images/{survey_id}.png",
    )

species = [column[10:] for column in species_columns[1:]]

print(" - submission.csv")
with open("processed_data/internal/submission.csv", "w") as f:
    f.write("surveyId,speciesId\n")
    for _, row in tqdm(internal_test_y_df.iterrows()):
        f.write(
            str(row[0])
            + ","
            + " ".join([species[i - 1] for i in range(1, len(row)) if row[i] == 1])
            + "\n"
        )
