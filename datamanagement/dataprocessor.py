import pandas as pd
from pvlib.location import lookup_altitude

pd.options.mode.chained_assignment = None   # Disable slice warning

class DataProcessor():

    def __init__(self, train: str = "data/GLC24_PA_metadata_train.csv", 
                       test:  str = "data/GLC24_PA_metadata_test.csv"
                ) -> None:
        self.train = pd.read_csv(train)
        self.test  = pd.read_csv(test) 

    def one_hot(self):
        

    
