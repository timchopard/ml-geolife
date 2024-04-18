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
        default="output/9-int-split-train.pkl",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="The path to the testing dataset.",
        default="output/9-int-split-test.pkl",
    )
    parser.add_argument(
        "--desc",
        type=str,
        help="A description of the data set to be used for the file path.",
        default="final",
    )
    return parser


args = init_argparse().parse_args()
assert os.path.exists(args.train), f"Training data file '{args.train}' not found."
assert os.path.exists(args.test), f"Testing data file '{args.test}' not found."

if args.train[-3:].lower() == "csv":
    train_df = pd.read_csv(args.train)
elif args.train[-3:].lower() == "pkl":
    train_df = pd.read_pickle(args.train)
else:
    raise ValueError("Invalid file format, must be either CSV or PKL.")

if args.test[-3:].lower() == "csv":
    test_df = pd.read_csv(args.test)
elif args.test[-3:].lower() == "pkl":
    test_df = pd.read_pickle(args.test)
else:
    raise ValueError("Invalid file format, must be either CSV or PKL.")

test_X_df = test_df.loc[:, ~test_df.columns.str.startswith("speciesId_")]
species_columns = ["surveyId"] + list(
    test_df.columns[test_df.columns.str.startswith("speciesId_")]
)
test_y_df = test_df.loc[:, species_columns]
test_y_df = test_y_df.drop(columns=["surveyId"])
test_y_count_df = test_y_df.sum(axis=1, numeric_only=True)

test_X_df = test_X_df.drop(columns=["surveyId"])
test_X_df.to_pickle("output/test_X.pkl")
test_y_df.to_pickle("output/test_y.pkl")
test_y_count_df.to_pickle("output/test_y_count.pkl")

train_X_df = train_df.loc[:, ~train_df.columns.str.startswith("speciesId_")]
species_columns = ["surveyId"] + list(
    train_df.columns[train_df.columns.str.startswith("speciesId_")]
)
train_X_df = train_X_df.drop(columns=["surveyId"])
train_y_df = train_df.loc[:, species_columns]
train_y_df = train_y_df.drop(columns=["surveyId"])
train_y_count_df = train_y_df.sum(axis=1, numeric_only=True)

train_X_df.to_pickle("output/train_X.pkl")
train_y_df.to_pickle("output/train_y.pkl")
train_y_count_df.to_pickle("output/train_y_count.pkl")

species = [column[10:] for column in species_columns[1:]]

with open("output/submission-actual.csv", "w") as f:
    f.write("surveyId,speciesId\n")
    for _, row in tqdm(test_y_df.iterrows()):
        f.write(
            str(row[0])
            + ","
            + " ".join([species[i - 1] for i in range(1, len(row)) if row[i] == 1])
            + "\n"
        )
