import pandas as pd
# import numpy as np
import sys

from pvlib.location import lookup_altitude

class DataLoader():
    """Loads in the data from specified file paths and preprocesses it
    """

    uniques = None

    
    def __init__(self, train: str, test: str, show=False):
        """Read in the train and test data
        """
        if train is not None:
            self.train = pd.read_csv(train)
            print(f":: Training data  with shape {self.train.shape} read in")
        if test is not None:
            self.test = pd.read_csv(test)
            print(f":: Test data with shape {self.train.shape} read in")
        else:
            print(":: No test data read in")
        if show:
            self._show_train_info(describe=True)

    def _show_train_info(self, detail=5, describe=False):
        print(self.train.head(detail))
        if describe:
            print(self.train.describe())
            print(self.train.info())


    def _drop_cols(self, col_list=["geoUncertaintyInM", "areaInM2",
                                   "region", "country"]):
        """Drop columns from the stored training data storing the new dataframe
        in the .cleaned member variable
        args:
        col_list    :   list    the columns to drop
        """
        self.cleaned = self.train.drop(col_list, axis=1)
        description =  ":: Dropped columns:\n::\t" + '\n::\t'.join(col_list)
        description += "\n:: New shape " + str(self.cleaned.shape)
        print(description)
    

    def _get_elevations(self):
        """Uses the pvlib to acquire elevations for each location based upon 
        the latitude and longtitude values. 
        """
        data = self._get_unique_locs
        self.uniques['elevation'] = self.uniques.apply(
                        lambda x: lookup_altitude(x['lat'], x['lon']), axis=1)
        unique_dict = dict(zip(self.uniques['loc_str'], self.uniques['elevation']))

    
    def _get_unique_locs(self):
        self.train['loc_str'] = self.train.apply(
            lambda x: str(x['lat']) + ',' + str(x['lon']), axis=1)
        return  self.train.drop_duplicates(subset="loc_str", keep=False)





# Run the data loader
def main():
    args = sys.argv[1:]
    print(args)
    data = None
    read_list = ['--read', '-r', '--elevation', '-e', '--uniques', '-u']
    if sum(1 for _ in [i for i in args if i in read_list]) > 0:
        path_train = "data/GLC24_PA_metadata_train.csv"
        data = DataLoader(path_train, None)
    if '--unique' in args or '-u' in args:
        data._get_unique_locs()
    if '--elevation' in args or '-e' in args:
        data._get_elevations(True)
    if data is not None:
        data._show_train_info()

        

if __name__ == "__main__":
    main()
