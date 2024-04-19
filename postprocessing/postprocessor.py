"""
"""

import re
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

class Postprocessor():
    """
    """

    def __init__(self, predictions = None, labels = None, pred_cols = None, 
                 pred_indices = None, **kwargs) -> None:
        """
        """
        if predictions.shape[0] != labels.shape[0]:
            Exception(":: [ERROR] Predictions and labels must have an equal "  \
                      "number of rows."
            )
        self.labels = labels
        self.predictions = predictions
        if type(predictions) is np.ndarray: 
            self.__preds_to_df(pred_cols, pred_indices)
        
        self.col_prefix = kwargs.pop("prefix") if "prefix" in kwargs else None
        self.col_suffix = kwargs.pop("suffix") if "suffix" in kwargs else None

    def normalise_columns(self):
        self.labels = self.labels.rename(
            columns=self.__format_columns(self.labels)
        )
        self.predictions = self.predictions.rename(
            columns=self.__format_columns(self.predictions)
        )
        
    def f1_score(self, **kwargs):
        """
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
        f1 = f1_score(self.predictions, self.labels, average='micro')
        if "print" in kwargs and kwargs.pop("print") is True:
            print(f":: [F1] {f1}")
        return f1

    def __format_columns(self, dataframe) -> dict:
        """
        """
        cols = {}
        for col in dataframe.columns:
            cols[col] = re.findall(r'\d+', col)[0] if type(col) is str else col
        return cols

    def __preds_to_df(self, columns, indices):
        """
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

