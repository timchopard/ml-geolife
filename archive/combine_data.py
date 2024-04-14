import pandas as pd
import numpy as np
import os

base_dir = "processed_data/internal_split/train"

# combined_df = pd.DataFrame()
# first = True
# for file in os.listdir(base_dir):
#     if file.endswith(".csv") and file != "metadata.csv":
#         print(f"Merging {file}")
#         tmp_df = pd.read_csv(os.path.join(base_dir, file)).drop(columns=["surveyId"])
#         columns = tmp_df.columns.map(lambda x: file[:-4]+"_"+x)
#         print(columns)
#         tmp_df.rename(columns=dict(zip(tmp_df.columns, columns)), inplace=True)
#         combined_df = pd.concat([combined_df, tmp_df], axis=1)


# with pd.option_context('mode.use_inf_as_na', True):
#     for column in combined_df.columns:
#         na_count = combined_df[column].isna().sum()
#         if na_count > 0:
#             print(f"{column}: {na_count} missing values")


