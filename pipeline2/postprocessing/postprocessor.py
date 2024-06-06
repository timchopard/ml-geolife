import pandas as pd
from datetime import datetime 
from .applycounts import apply_counts

class GeoLifePostprocessor():

    def __init__(self, predictions, weights, counts):
        self.predictions = None 
        for prediction, weight in zip(predictions, weights):
            prediction = prediction.to_numpy()
            if self.predictions is None: 
                self.predictions = prediction * weight 
            else:
                self.predictions += prediction * weight 
        self.predictions = apply_counts(self.predictions, counts)
        self.predictions = pd.DataFrame(
            self.predictions, 
            index=predictions[0].index, 
            columns=predictions[0].columns
        )

    def save(self, filename:str = ""):
        filename += str(datetime.now()).replace(' ', '_').replace(':', '-', 1)
        filename = f"{filename.rsplit(':', maxsplit=1)[0]}.csv"
        self.__format_submission().to_csv(
            f"submissions/{filename}", 
            index=False
        )
    
    def __format_submission(self):
        output_list = []
        for _, row in self.predictions.iterrows():
            row_string = ""
            for truth, value in zip(row, self.predictions.columns):
                row_string += str(value.split('_')[-1]) + " " if truth else ""
            output_list.append(row_string)

        return pd.DataFrame({
            "surveyId"      :   self.predictions.index,
            "predictions"   :   output_list
        })