from PIL import Image
from tqdm import tqdm
from pathlib import Path
import pandas as pd
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
        default="output/3-m-tidy_train.pkl",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="The path to the testing dataset.",
        default="_raw/GLC24_PA_metadata_test.csv",
    )
    parser.add_argument(
        "--desc",
        type=str,
        help="A description of the data set to be used for the file path.",
        default="ohe",
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

print("One-hot encode country, region, and speciesId in training data")
for column in ["country", "region", "speciesId"]:
    print(f" - Processing: {column}")
    train_df = pd.concat(
        [train_df, pd.get_dummies(train_df[column], prefix=column)],
        axis=1,
    )
    train_df = train_df.drop(columns=[column])

print("One-hot encode country and region in test data")
for column in ["country", "region"]:
    print(f" - Processing: {column}")
    test_df = pd.concat(
        [test_df, pd.get_dummies(test_df[column], prefix=column)],
        axis=1,
    )
    test_df = test_df.drop(columns=[column])

train_df.to_pickle(f"output/4-{args.desc}_train.pkl")
print(f"4-{args.desc}_train.pkl has shape {train_df.shape}")
test_df.to_pickle(f"output/4-{args.desc}_test.pkl")
print(f"4-{args.desc}_test.pkl has shape {test_df.shape}")
