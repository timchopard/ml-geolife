""" from pipeline.preprocess import PCACreator, PCAApplier
"""
import os
import pickle as pk
import pandas as pd
from warnings import simplefilter
from numpy import float32, float64, inf, nan
from sklearn.preprocessing import scale

# Hide irrelevant "performance wanrning" from pandas library 
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

class PCAGeneric():
    """ Generic class providing methods and variables to PCACreator and 
    PCAApplier classes
    """

    rw_path = "models/pca"

    high_cov_hf_cols = [
        'HumanFootprint-Built1994', 
        'HumanFootprint-Built2009', 
        'HumanFootprint-croplands1992', 
        'HumanFootprint-croplands2005', 
        'HumanFootprint-Lights1994', 
        'HumanFootprint-Lights2009', 
        'HumanFootprint-Pasture1993', 
        'HumanFootprint-Pasture2009', 
        'HumanFootprint-Popdensity1990', 
        'HumanFootprint-Popdensity2010', 
        'HumanFootprint-Railways',
    ]
    drop_cols = [
        "HumanFootprint-HFP1993", 
        "HumanFootprint-NavWater1994", 
        "HumanFootprint-Roads",
        "Latvia",
    ]
    tas_prefixes = [
        "Bio-tas_",
        "Bio-tasmax_",
        "Bio-tasmin_",
    ]
    pr_prefix = [
        "Bio-pr_",
    ]

    def __init__(self, path, **kwargs):
        path = os.path.join("data/processed", path + "_full.pkl")
        self.data = pd.read_pickle(path)
        if "from_year" in kwargs:
            self.from_year = kwargs.pop("from_year")
            self.__drop_years()
        

    def apply_human_footprint(self):
        """Applies the PCA model for human footprint, combining several 
        columns and deleting the originals
        """
        pca = self._load("pca_hf")
        self.data["HumanFootprintGeneric"] = pca.transform(
            self.data[self.high_cov_hf_cols]
        )
        self.data.drop(columns=self.high_cov_hf_cols, inplace=True)

    def apply_rainfalls(self):
        """Applies the PCA model for rainfall and temperature, combining several 
        columns and deleting the originals
        """
        self._process_rainfall(
            self.__apply_rainfall_helper,
            self.tas_prefixes
        )
        self._process_rainfall(
            self.__apply_rainfall_helper,
            self.pr_prefix
        )

    def _process_rainfall(self, action, prefixes):
        """ Loops through the months and years surveyed to pass column names
        to the relevant function

        args:
            action          :   The function to apply to each set of columns
            prefixes        :   The column name prefixes to focus on
        """
        for month in range(1, 13):
            combination_cols = []
            for year in range(self.from_year, 2019):
                suffix = f"{'0' if month < 10 else ''}{month}_{year}"
                for name in prefixes:
                    combination_cols.append(f"{name}{suffix}")

            filename = f"pca_{'BioTas' if len(prefixes) == 3 else 'BioPr'}_"
            filename = f"{filename}{'0' if month < 10 else ''}{month}"
            action(combination_cols, filename)

    def _remove_nan(self):
        """ Removes NaN values from the data, replacing them with the median
        values for their respective columns.
        """
        self.data.replace([inf, -inf], nan, inplace=True)
        self.data = self.data.fillna(self.data.median())

    def _load(self, filename):
        """ Loads in a stored PCA model
        """
        return pk.load(
            open(os.path.join(self.rw_path, filename + ".pkl"), 'rb')
        )
        # TODO: add try/except

    def _reduce_size(self):
        """ Reduces the float64 values to float32 and drops unneeded columns to
        reduce the memory usage
        """
        mask = (self.data.dtypes == float64).tolist()
        for col, is_masked in zip(self.data.columns, mask):
            if is_masked:
                self.data[col] = self.data[col].astype(float32)
        for col in self.drop_cols:
            if col in self.data.columns.to_list():
                self.data.drop(columns=col, inplace=True)

    def __apply_rainfall_helper(self, col_names, filename):
        """ Helper function for apply_rainfall. Loads in PCA models and applies
        them to the relevant columns, deletes the original columns
        """
        pca = self._load(filename)
        self.data[filename] = pca.transform(self.data[col_names])
        self.data.drop(columns=col_names, inplace=True)

    def __drop_years(self):
        for year in range(2000, self.from_year):
            self.data = self.data[
                [col for col in self.data if '_' + str(year) not in col        \
                 or "species" in col]
            ]


class PCACreator(PCAGeneric):
    from sklearn.decomposition import PCA

    def __init__(self, path=None, **kwargs) -> None:
        super().__init__(path, **kwargs)
        self.__kwarg_handler(**kwargs)

    def human_footprint(self):
        pca = self.PCA(n_components=1)
        pca.fit(self.data[self.high_cov_hf_cols])
        self.__save("pca_hf", pca)
        self.apply_human_footprint()

    def rainfalls(self):
        self._process_rainfall(
            self.__combine_rainfall_helper,
            self.tas_prefixes
        )
        self._process_rainfall(
            self.__combine_rainfall_helper,
            self.pr_prefix
        )
        self.apply_rainfalls()

    def __combine_rainfall_helper(self, col_list, filename):
        pca = self.PCA(n_components=1)
        pca.fit(self.data[col_list])
        self.__save(filename, pca)

    def __build_filepath(self):
        for dir in ["models/", "models/pca/"]:
            if not os.path.exists(dir):
                print(f":: Creating directory {dir}")
                os.mkdir(dir)

    def __kwarg_handler(self, **kwargs):
        if "method" in kwargs:
            method = kwargs.pop("method")

            if method in ["preprocess", "full"]:
                self._remove_nan()
                self._reduce_size()

            if method == "full":
                self.human_footprint()
                self.rainfalls()
                self.__save_cols()

    def __save(self, filename, model):
        self.__build_filepath()
        pk.dump(
            model,
            open(os.path.join(self.rw_path, filename + ".pkl"), "wb")
        )

    def __save_cols(self):
        columns = self.data.columns.to_list()
        columns = [col for col in columns if "species" not in col]
        self.__save("final_cols", columns)


class PCAApplier(PCAGeneric):

    def __init__(self, path=None, **kwargs):
        super().__init__(path, **kwargs)
        self.__kwarg_handler(**kwargs)
        self.xx_cols = [col for col in self.data if "species" not in col]
        self.yy_cols = [col for col in self.data if "species" in col]

    def __kwarg_handler(self, **kwargs):
        if "method" in kwargs:
            method = kwargs.pop("method")

            if method in ["full",]:
                self._remove_nan()

            if method in ["full", "notnan"]:
                self._reduce_size()
                self.apply_human_footprint()
                self.apply_rainfalls()
                self.__add_cols()

    def __add_cols(self):
        true_cols = self._load("final_cols")
        new_cols = list(set(true_cols) - set(self.data.columns))
        self.data[new_cols] = False
        if self.data.shape[1] == len(true_cols):
            self.data = self.data[true_cols]
