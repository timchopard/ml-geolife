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
        default="output/1-top-500_train.pkl",
    )
    parser.add_argument(
        "--desc",
        type=str,
        help="A description of the data set to be used for the file path.",
        default="m-tidy",
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

drop_regions = {
    "BLACK SEA",
    "PANNONIAN",
}
train_df = train_df[~train_df.region.isin(drop_regions)]

train_df.surveyId = train_df.surveyId.astype(int)
train_df.speciesId = train_df.speciesId.astype(int)

print(f"{train_df.shape=}")

train_df.to_pickle(f"output/3-{args.desc}_train.pkl")
print(f"3-{args.desc}_train.pkl has shape {train_df.shape}")
