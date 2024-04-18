import pandas as pd
import argparse as ap
import os
from pathlib import Path


def init_argparse() -> ap.ArgumentParser:
    parser = ap.ArgumentParser(
        usage="%(prog)s [OPTION]",
        description="Filter the top N species from the training dataset.",
    )
    parser.add_argument(
        "--number",
        "-n",
        type=int,
        help="The number of species to retain.",
        default=500,
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="The path to the training dataset.",
        default="_raw/GLC24_PA_metadata_train.csv",
    )
    return parser


args = init_argparse().parse_args()
assert args.number > 0, f"Invalid number of species, got {args.number}."
assert os.path.exists(args.file), f"File '{args.file}' not found."

if args.file[-3:] == "csv":
    train_df = pd.read_csv(args.file)
elif args.file[-3:] == "pkl":
    train_df = pd.read_pickle(args.file)
else:
    raise ValueError("Invalid file format, must be either CSV or PKL.")

assert "speciesId" in train_df.columns, "Column 'speciesId' not found."

top_n = sorted(
    list(train_df.speciesId.value_counts().head(args.number).index.astype(int))
)
top_n_df = train_df[train_df.speciesId.isin(top_n)]

top_n_df.loc[:, "region"] = top_n_df.region.astype("category")
top_n_df.loc[:, "country"] = top_n_df.country.astype("category")
top_n_df.loc[:, "speciesId"] = top_n_df.speciesId.astype(int)

print(f"Original data: {train_df.shape}\nTop {args.number} species: {top_n_df.shape}")

Path("output").mkdir(parents=True, exist_ok=True)
top_n_df.to_pickle(f"output/1-top-{args.number}_train.pkl")
print(f"1-top-{args.number}_train.pkl has shape {top_n_df.shape}")
