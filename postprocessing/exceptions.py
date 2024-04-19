"""Exceptions and errors for the postprocessor
"""

class ShapeError(Exception):
    """An error for when the predictions and labels do not have the same number 
    of columns
    """
    message = ":: [ERROR] Predictions and labels must have an equal number of "\
              "rows. If this is for a submission run again with uploadable=True"

    def __init__(self) -> None:
        """Passes the message directly to the Exception class
        """
        super().__init__(self.message)


class MissingDataError(Exception):
    """An error when insufficient data has been passed to construct the 
    predictions dataframe
    """

    def __init__(self, columns, indices) -> None:
        """Passes the message directly to the Exception class
        """
        super().__init__(self.__write_message(columns, indices))

    def __write_message(self, columns, indices):
        """Constructs the error message

        args:
            columns     :   The data passed in for the column names
            indices     :   The data passed in for the indice values
        """
        col_section = ":: Missing pred_cols argument.\n-\tPlease pass a list " \
                    + "of column values. These could be taken from the "       \
                    + "pd.DataFrame.columns value of the training labels.\n"
        idx_section = ":: Missing pred_indices argument.\n-\tPlease pass a "   \
                    + "list of index values. These could be taken from the "   \
                    + "pd.DataFrame.index value of the test input data\n"
        return col_section if not isinstance(columns, list) else "" +          \
               idx_section if not isinstance(indices, list) else ""
