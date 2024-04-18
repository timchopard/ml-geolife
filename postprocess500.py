import pandas as pd
import numpy as np

class Postprocessor():


    def __init__(self, symbiosis_path:str = "symbiosis.csv") -> None:
        """Reads in the symbiote data from the provided path. Loads the top
        500 species from the raw CSV PA data.

        args:
            symbiosis_path      :   The path to the symbiosis data
        """
        self.symbiosis = pd.read_csv(symbiosis_path)
        self.top_500 = list(pd.read_csv(
            "data/GLC24_PA_metadata_train.csv"
        )["speciesId"].value_counts().to_dict().keys())
        self.top_500 = [int(x) for x in self.top_500][:500]

    def add_symbiotes(self, threshold:float = 0.8):
        """
        """
        reduced = self.symbiosis = self.symbiosis[(                            
            self.symbiosis['occurrence'] > threshold
        )]
        reduced = reduced[reduced["parent"].isin(self.top_500)]
        reduced = reduced[~reduced["child"].isin(self.top_500)]

        bayes



