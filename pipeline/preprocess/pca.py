import os
import pickle as pk
import pandas as pd
from numpy import float32, float64, inf, nan
from sklearn.preprocessing import scale 

class PCAGeneric():

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

    def apply_human_footprint(self):
        pca = self._load("pca_hf")
        self.data["HumanFootprintGeneric"] = pca.transform(
            self.data[self.high_cov_hf_cols]
        )
        self.data.drop(columns=self.high_cov_hf_cols, inplace=True)

    def apply_rainfalls(self):
        self._process_rainfall(
            self.__apply_rainfall_helper,
            self.tas_prefixes
        )
        self._process_rainfall(
            self.__apply_rainfall_helper,
            self.pr_prefix
        )

    def _process_rainfall(self, action, prefixes):
        for month in range(1, 13):
            combination_cols = []
            for year in range(2000, 2019):
                suffix = f"{'0' if month < 10 else ''}{month}_{year}"
                for name in prefixes:
                    combination_cols.append(f"{name}{suffix}")

            filename = f"pca_{'BioTas' if len(prefixes) == 3 else 'BioPr'}_"
            filename = f"{filename}{'0' if month < 10 else ''}{month}"
            action(combination_cols, filename)

    def _remove_nan(self):
        self.data.replace([inf, -inf], nan, inplace=True)
        self.data = self.data.fillna(self.data.median())

    def _load(self, filename):
        return pk.load(
            open(os.path.join(self.rw_path, filename + ".pkl"), 'rb')
        )        
        # TODO: add try/except           

    def _reduce_size(self):
        mask = (self.data.dtypes == float64).tolist()
        for col, is_masked in zip(self.data.columns, mask):
            if is_masked:
                self.data[col] = self.data[col].astype(float32)
        for col in self.drop_cols:
            if col in self.data.columns.to_list():
                self.data.drop(columns=col, inplace=True)

    def __apply_rainfall_helper(self, col_names, filename):
        pca = self._load(filename)
        self.data[filename] = pca.transform(self.data[col_names])
        self.data.drop(columns=col_names, inplace=True)

class PCACreator(PCAGeneric):
    from sklearn.decomposition import PCA 

    def __init__(self, path=None, **kwargs) -> None:
        path = "data/processed/train_full.pkl" if path is None else path
        self.data = pd.read_pickle(path)
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
        path = "data/processed/test_full.pkl" if path is None else path
        self.data = pd.read_pickle(path)
        self.__kwarg_handler(**kwargs)


    def __kwarg_handler(self, **kwargs):
        if "method" in kwargs:
            method = kwargs.pop("method")

            if method in ["full",]:
                self._remove_nan()
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
                
