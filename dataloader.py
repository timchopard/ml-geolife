import pandas as pd
from pvlib.location import lookup_altitude

class DataLoader():
    """Loads in the data from specified file paths and preprocesses it
    """

    def __init__(self, train: str, test: str):
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
        data = self._get_unique_locs()
        print(f":: Unique locations: {data.shape[0]}")
        data['elevation'] = data.apply(
                        lambda x: lookup_altitude(x['lat'], x['lon']), axis=1)
        print(f":: Found elevations for each location")
        unique_dict = dict(zip(data['loc_str'], data['elevation']))
        self.train['elevation'] = self.train['loc_str'].map(unique_dict)
        self.train.drop(columns=["loc_str"], inplace=True)
        print(self.train.head())


    def _get_unique_locs(self):
        """Generates a subset of the data based on each unique longtitude and 
        lattitude pair (note: the number of unique pairs is slightly lower than
        the number of unique surveyId values, as the same location has been used
        twice for some surveys).
        """
        self.train['loc_str'] = self.train.apply(
            lambda x: str(x['lat']) + ',' + str(x['lon']), axis=1)
        return self.train.drop_duplicates(subset="loc_str")
