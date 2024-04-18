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
        "--test",
        type=str,
        help="The path to the testing dataset.",
        default="_raw/GLC24_PA_metadata_test.csv",
    )
    parser.add_argument(
        "--desc",
        type=str,
        help="A description of the data set to be used for the file path.",
        default="all",
    )
    return parser


args = init_argparse().parse_args()
assert os.path.exists(args.train), f"Training data file '{args.train}' not found."
assert os.path.exists(args.test), f"Testing data file '{args.test}' not found."

if args.train[-3:].lower() == "csv":
    train_survey_ids = pd.read_csv(args.train).surveyId.unique().astype(str)
elif args.train[-3:].lower() == "pkl":
    train_survey_ids = pd.read_pickle(args.train).surveyId.unique().astype(str)
else:
    raise ValueError("Invalid file format, must be either CSV or PKL.")

if args.test[-3:].lower() == "csv":
    test_survey_ids = pd.read_csv(args.test).surveyId.unique().astype(str)
elif args.test[-3:].lower() == "pkl":
    test_survey_ids = pd.read_pickle(args.test).surveyId.unique().astype(str)
else:
    raise ValueError("Invalid file format, must be either CSV or PKL.")

train_path = Path(f"output/2-{args.desc}_images") / "train"
test_path = Path(f"output/2-{args.desc}_images") / "test"
Path(train_path).mkdir(parents=True, exist_ok=True)
Path(test_path).mkdir(parents=True, exist_ok=True)

for set_type, survey_ids, path in [
    ("train", train_survey_ids, train_path),
    ("test", test_survey_ids, test_path),
]:
    for survey_id in tqdm(survey_ids):
        cd = survey_id[-4:-2]
        ab = survey_id[-2:]
        rgb = Image.open(
            f"_raw/PA_{set_type.capitalize()}_SatellitePatches_RGB/pa_{set_type}_patches_rgb/{ab}/{cd}/{survey_id}.jpeg"
        )
        nir = Image.open(
            f"_raw/PA_{set_type.capitalize()}_SatellitePatches_NIR/pa_{set_type}_patches_nir/{ab}/{cd}/{survey_id}.jpeg"
        ).convert("L")
        rgb.putalpha(nir)
        rgb.save(path / f"{survey_id}.png")

print(f"{len(list(train_path.glob('*')))} images saved to {train_path}")
print(f"{len(list(test_path.glob('*')))} images saved to {test_path}")
