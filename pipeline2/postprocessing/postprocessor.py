import numpy as np
import pickle as pk
import pandas as pd
from datetime import datetime 
from .applycounts import apply_counts

class GeoLifeOutputLoader():

    def __init__(self, species, surveys):
        self.species = species 
        self.surveys = surveys 

    def load_outputs(self, path_list:list):
        outputs = []
        for path in path_list:
            loaded_data = (pd.read_pickle(path))
            if type(loaded_data) == np.ndarray:
                loaded_data = pd.DataFrame(
                    loaded_data,
                    columns=self.species,
                    index=self.surveys
                )
            else:
                loaded_data = loaded_data[self.species]
                loaded_data = loaded_data.set_index(self.surveys)
            outputs.append(loaded_data)
        return outputs
    
    def load_counts(self, path):
        return pk.load(open(path, "rb"))


class GeoLifePostprocessor():

    def __init__(self, predictions, weights, counts):
        self.index = predictions[0].index
        self.predictions = None 
        for prediction, weight in zip(predictions, weights):
            pred = prediction.to_numpy()
            if self.predictions is None: 
                self.predictions = pred * weight 
            else:
                self.predictions += pred * weight 
        pd.DataFrame(self.predictions).to_csv("test-new.csv", index=False)
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
        print(f"Saved to: submissions/{filename}")
    
    def __format_submission(self):
        output_list = []
        for _, row in self.predictions.iterrows():
            row_string = ""
            for truth, value in zip(row, self.predictions.columns):
                row_string += str(value.split('_')[-1]) + " " if truth else ""
            output_list.append(row_string)

        return pd.DataFrame({
            "surveyId"      :   self.index,
            "predictions"   :   output_list
        })