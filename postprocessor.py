import pandas as pd
import pickle

class Postprocessor():

    def __init__(self, predictions, id_keys,
                 species_path:str = "processed_data/postprocess-speices"):
        self.predictions = predictions
        self.id_keys = id_keys 
        with open(species_path, 'rb') as fp:
            self.species = pickle.load(fp)

    def process(self):
        output_list = []
        for _, row in self.predictions.iterrows():
            row_string = ""
            for truth, value in zip(row, self.species):
                if truth:
                    row_string += str(value) + " "
            output_list.append(row_string)

        return pd.DataFrame({
            "surveyId"      : self.id_keys,
            "predictions"   : output_list
        })
    
    def save(self, filename:str):
        output = self.process()
        output.to_csv(filename, index=False)