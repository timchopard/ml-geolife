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
import argparse as ap
import os


def init_argparse() -> ap.ArgumentParser:
    parser = ap.ArgumentParser(
        usage="%(prog)s [OPTION]",
        description="Convert the satellite images to PNG format.",
    )
    parser.add_argument(
        "--train",
        type=str,
        help="The path to the training dataset.",
        default="output/8-tidy_train.pkl",
    )
    parser.add_argument(
        "--desc",
        type=str,
        help="A description of the data set to be used for the file path.",
        default="int-split",
    )
    return parser


args = init_argparse().parse_args()
assert os.path.exists(args.train), f"Training data file '{args.train}' not found."

if args.train[-3:].lower() == "csv":
    train_df = pd.read_csv(args.train)
elif args.train[-3:].lower() == "pkl":
    train_df = pd.read_pickle(args.train)
else:
    raise ValueError("Invalid file format, must be either CSV or PKL.")


internal_train_df = train_df.sample(frac=0.8, random_state=113)
internal_test_df = train_df.drop(internal_train_df.index)

print(f"{internal_train_df.shape=}")
print(f"{internal_test_df.shape=}")


internal_train_df.to_pickle(f"output/9-{args.desc}-train.pkl")
print(f"9-{args.desc}-train.pkl has shape {internal_train_df.shape}")

Path("output/9-internal-train-images").mkdir(parents=True, exist_ok=True)
for survey_id in tqdm(internal_train_df.surveyId):
    shutil.copy(
        f"output/2-all_images/train/{survey_id}.png",
        f"output/9-internal-train-images/{survey_id}.png",
    )

Path("output/9-internal-test-images").mkdir(parents=True, exist_ok=True)
for survey_id in tqdm(internal_train_df.surveyId):
    shutil.copy(
        f"output/2-all_images/train/{survey_id}.png",
        f"output/9-internal-test-images/{survey_id}.png",
    )

internal_test_df.to_pickle(f"output/9-{args.desc}-test.pkl")
print(f"9-{args.desc}-test.pkl has shape {internal_test_df.shape}")
