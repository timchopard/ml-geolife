"""
# from postprocessing.postprocessor import Postprocessor

Postprocessor for prediction data. This is designed to be called from notebooks
in the root directory.

    - Converts data to dataframe of booleans with given index and column names
    - Performs f1 score on the data
    - Formats submission and saves it to the 'submissions' directory
"""

import os
import re
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

class Postprocessor():
    """
    """

    def __init__(self, predictions, labels, pred_cols = None, 
                 pred_indices = None, **kwargs) -> None:
        """
        Converts the predictions to a DataFrame and parses kwargs

        args:
            predictions     :   Either a numpy array or a pandas DataFrame of 
                                predicted values
            labels          :   A pandas DataFrame of the labels corresponding 
                                to the predictions
            pred_cols       :   [Not needed if predictions is a DataFrame] 
                                A list of the column names for the predictions
            pred_indices    :   [Not needed if predictions is a DataFrame] 
                                A list of the indice values for the predictions

        kwargs:
            f1              :   True to perform a f1 score immediately
            save            :   True to save to default submission path                    
            model_type      :   The type of model for the saved filename
            uploadable      :   True if the predictions are from the kaggle
                                test data
        """
        if predictions.shape[0] != labels.shape[0]:
            Exception(":: [ERROR] Predictions and labels must have an equal "  \
                      "number of rows."
            )
        self.labels = labels
        self.predictions = predictions
        if type(predictions) is np.ndarray: 
            self.__preds_to_df(pred_cols, pred_indices)

        self.__normalise_columns()
        self.__parse_kwargs(**kwargs)


    def f1_score(self, **kwargs) -> np.float64:
        """Formats the predictions dataframe so that it matches the labels 
        dataframe in structure then calculates and prints the f1 score to three
        decimal places. Stores the untruncated value as a member variable.
        """
        # Fill in any missing columns in the predictions
        self.predictions = pd.concat(
            (self.predictions, pd.DataFrame(
                columns=list(
                    set(self.labels.columns) - set(self.predictions.columns)
                ),
                index = self.predictions.index,
                dtype=np.int8
            )), axis=1
        ).fillna(0).astype(bool)

        # Reindex the predictions to match the labels
        self.predictions = self.predictions.reindex(columns=self.labels.columns)

        # Get f1 score
        self.f1 = f1_score(self.predictions, self.labels, average='micro')
        
        if "print" in kwargs and kwargs.pop("print") == True:
            print(f":: [F1] {self.f1:.3f}")

    def save_predictions(self, model_type:str = None) -> None:
        """Creates a submissions directory if none exists. Saves the formatted
        dataframe in the submissions directory with a filename based on the 
        model used and the current time.

        args:
            model_type      :   The name of the model used to provide the start
                                of the filename. If ommitted the file will be
                                saved with only a timestamp.
        """
        dir_path = "submissions"
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        # Create a filename based on the model type 
        # and the current date and time
        filename = str(datetime.now()).replace(' ', '_')
        filename = f"{filename.replace(':', '-', 1).split(':')[0]}.csv"
        if model_type is not None: filename = f"{model_type}_{filename}"
        if self.uploadable: filename = f"uploadable_{filename}"
        
        # Format the final output and save it to a .csv file 
        self.__format_submission().to_csv(
            os.path.join(
                dir_path, filename 
        ), index=False)

    def __format_submission(self) -> pd.DataFrame:
        """Formats the predictions dataframe for submitting. 

        return:
            pd.DataFrame    :   The formatted dataframe with only surveyId and
                                predictions columns
        """
        output_list = []
        for _, row in self.predictions.iterrows():
            row_string = ""
            for truth, value in zip(row, self.predictions.columns):
                if truth: row_string += str(value) + " "
            output_list.append(row_string)
        
        return pd.DataFrame({
            "surveyId"      :   self.predictions.index,
            "predictions"   :   output_list
        })

    def __format_columns(self, dataframe) -> dict:
        """Helper function
        Strips any prefixes or suffixes from the speciesId column headers

        args:
            dataframe       :   The dataframe to work on

        return: 
            list            :   A list of the formatted column headers
        """
        cols = {}
        for col in dataframe.columns:
            cols[col] = re.findall(r'\d+', col)[0] if type(col) is str else col
        return cols

    def __preds_to_df(self, columns, indices):
        """Converts a numpy array of predictions to a pandas DataFrame

        args:
            columns         :   The column names (speciesId)
            indices         :   The index names (surveyId)
        """
        if type(columns) is not list or type(indices) is not list:
            col_section = ":: Missing pred_cols argument.\n" \
                        + "-\tPlease pass a list of column values. These could"\
                        + " be taken from the pd.DataFrame.columns value of"   \
                        + " the training labels.\n"
            idx_section = ":: Missing pred_indices argument.\n"                \
                        + "-\tPlease pass a list of index values. These could" \
                        + " be taken from the pd.DataFrame.index value of the" \
                        + " test input data\n"
            Exception(
                col_section if type(columns) is not list else "" +
                idx_section if type(indices) is not list else ""
            )
        self.predictions = pd.DataFrame(
            self.predictions, 
            columns=columns,
            index=indices
        )

    def __normalise_columns(self):
        """Uses the __format_columns helper function to rename all columns as 
        only the speciesId values without prefixes or suffixes
        """
        self.labels = self.labels.rename(
            columns=self.__format_columns(self.labels)
        )
        self.predictions = self.predictions.rename(
            columns=self.__format_columns(self.predictions)
        )
        
    def __parse_kwargs(self, **kwargs):
        """Helper function
        Parses key word arguments for the initialiser
        """
        model = kwargs.pop("model_type") if "model_type" in kwargs else None
        
        self.uploadable = (
            "uploadable" in kwargs and kwargs.pop("uploadable") == True 
        ) 

        if "f1" in kwargs and kwargs.pop("f1") == True:
            self.f1_score(print=True)

        if "save" in kwargs and kwargs.pop("save") == True:
            self.save_predictions(model)
