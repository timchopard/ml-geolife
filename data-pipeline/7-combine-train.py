import pandas as pd
import argparse as ap
import os


def init_argparse() -> ap.ArgumentParser:
    parser = ap.ArgumentParser(
        usage="%(prog)s [OPTION]",
        description="Convert the satellite images to PNG format.",
    )
    parser.add_argument(
        "--metadata",
        "-m",
        type=str,
        help="The path to the metadata.",
        default="output/5-grp_train.pkl",
    )
    parser.add_argument(
        "--additions",
        "-a",
        type=str,
        nargs="+",
        help="The path to the additional datasets.",
        default=[
            "_raw/EnvironmentalRasters/EnvironmentalRasters/Climate/Average 1981-2010/GLC24-PA-train-bioclimatic.csv",
            # "_raw/EnvironmentalRasters/EnvironmentalRasters/Climate/Monthly/GLC24-PA-train-bioclimatic_monthly.csv",
            "_raw/EnvironmentalRasters/EnvironmentalRasters/Elevation/GLC24-PA-train-elevation.csv",
            "_raw/EnvironmentalRasters/EnvironmentalRasters/Human Footprint/GLC24-PA-train-human_footprint.csv",
            "_raw/EnvironmentalRasters/EnvironmentalRasters/LandCover/GLC24-PA-train-landcover.csv",
            "_raw/EnvironmentalRasters/EnvironmentalRasters/SoilGrids/GLC24-PA-train-soilgrids.csv",
            # "_raw/PA-train-landsat_time_series/GLC24-PA-train-landsat_time_series-blue.csv",
            # "_raw/PA-train-landsat_time_series/GLC24-PA-train-landsat_time_series-green.csv",
            # "_raw/PA-train-landsat_time_series/GLC24-PA-train-landsat_time_series-nir.csv",
            # "_raw/PA-train-landsat_time_series/GLC24-PA-train-landsat_time_series-red.csv",
            # "_raw/PA-train-landsat_time_series/GLC24-PA-train-landsat_time_series-swir1.csv",
            # "_raw/PA-train-landsat_time_series/GLC24-PA-train-landsat_time_series-swir2.csv",
        ],
    )
    parser.add_argument(
        "--desc",
        type=str,
        help="A description of the data set to be used for the file path.",
        default="train",
    )
    return parser


args = init_argparse().parse_args()
assert os.path.exists(args.metadata), f"File '{args.metadata}' not found."
for file in args.additions:
    assert os.path.exists(file), f"File '{file}' not found."

if args.metadata[-3:].lower() == "csv":
    combined_df = pd.read_csv(args.metadata)
elif args.metadata[-3:].lower() == "pkl":
    combined_df = pd.read_pickle(args.metadata)
else:
    raise ValueError("Invalid file format, must be either CSV or PKL.")

for idx, file in enumerate(args.additions):
    if file[-3:].lower() == "csv":
        tmp_df = pd.read_csv(file)
    elif file[-3:].lower() == "pkl":
        tmp_ = pd.read_pickle(file)
    else:
        raise ValueError("Invalid file format, must be either CSV or PKL.")
    print(f"Processing train data: {file}")
    print(f" - {combined_df.shape}")
    combined_df = pd.merge(
        combined_df, tmp_df, on="surveyId", suffixes=("", f"_{chr(ord('a') + idx)}")
    )

combined_df.to_pickle(f"output/7-combined_{args.desc}.pkl")
print(f"7-combined_{args.desc}.pkl has shape {combined_df.shape}")
