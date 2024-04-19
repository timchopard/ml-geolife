"""Verifies that the processed test data is valid.
"""

import os
import pandas as pd

def verify_test(dataframe:pd.DataFrame,
                dir_path:str = "data-pipeline/_raw") -> bool:
    """Verify that the surveyId values in the processed data match those in the
    raw test data.
    """
    processed_cols = dataframe["surveyId"] if "surveyId" in dataframe.columns \
                                         else dataframe.index
    original_cols = pd.read_csv(
        os.path.join(dir_path, "GLC24_PA_metadata_test.csv")
    )["surveyId"]

    only_in_proc = list(set(processed_cols) - set(original_cols))
    only_in_orig = list(set(original_cols) - set(processed_cols))

    message = ""
    if len(only_in_proc) > 0:
        message += f":: {len(only_in_proc)} IDs only in the processed data.\n"
    if len(only_in_orig) > 0:
        message += f":: {len(only_in_orig)} IDs only in the raw data.\n"
    if len(message) > 0:
        print(message)

    return len(only_in_orig) + len(only_in_proc) == 0
