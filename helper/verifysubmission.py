"""Verifies that the contents of a submission file appears to be correct
"""

import os
import pandas as pd

def verify_submission(filename:str,
                      raw_path:str = "data-pipeline/_raw") -> bool:
    """Checks that the shape and surveyId values of a submission CSV file are
    correct

    args:
        filename        :   The name of the submission file to be checked
        raw_path        :   The directory in which the sample submission is
                            located

    return:
        bool            :   True if the file appears correct, otherwise False
    """
    sample_submission = pd.read_csv(
        os.path.join(raw_path, "GLC24_SAMPLE_SUBMISSION.csv")
    )
    generated_submission = pd.read_csv(os.path.join("submissions", filename))

    if sample_submission.shape != generated_submission.shape:
        print(":: Shape of dataframes is not equal.")
        return False

    correct_ids = True
    for sample_id in sample_submission["surveyId"]:
        if sample_id not in generated_submission["surveyId"].to_list():
            print(f":: ID {sample_id} in sample but not predictions")
            correct_ids = True
    for prediction_id in generated_submission["surveyId"]:
        if prediction_id not in sample_submission["surveyId"].to_list():
            print(f":: ID {prediction_id} in predictions but not in sample")
            correct_ids = False

    return correct_ids
