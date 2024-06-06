import pandas as pd
from sklearn.decomposition import PCA
from numpy import inf, nan, float32, float64
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

class GeoLifePCA():
    __high_cov_hf_cols = [
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
    __drop_cols = [
        "HumanFootprint-HFP1993", 
        "HumanFootprint-HFP2009",
        "HumanFootprint-NavWater1994", 
        "HumanFootprint-NavWater2009",
        "HumanFootprint-Roads",
        "Latvia",
    ]
    __tas_prefixes = [
        "Bio-tas_",
        "Bio-tasmax_",
        "Bio-tasmin_",
    ]
    __pr_prefix = [
        "Bio-pr_",
    ]

    def __init__(self, train, test):
        self.train = train 
        self.test = test

    def process_data(self):
        self.__remove_nan()
        self.train = self.__reduce_size(self.train)
        self.test = self.__reduce_size(self.test)
        self.__apply_pca(self.__high_cov_hf_cols, "pca_hf")
        self.__process_rainfall(self.__tas_prefixes)
        self.__process_rainfall(self.__pr_prefix)
        self.__final_drop()
        self.__balance_test_cols()
        return self.train, self.test

    def __remove_nan(self):
        self.train.replace([inf, -inf], nan, inplace=True)
        self.train.replace([inf, -inf], nan, inplace=True)
        self.test = self.test.fillna(self.test.median())
        self.test = self.test.fillna(self.test.median())

    def __reduce_size(self, data):
        mask = (data.dtypes == float64).tolist()
        for col, is_masked in zip(data.columns, mask):
            if is_masked:
                data[col] = data[col].astype(float32)
        for col in self.__drop_cols:
            if col in data.columns.to_list():
                data.drop(columns=col, inplace=True)
        return data

    def __apply_pca(self, columns, new_col_name):
        pca = PCA(n_components=1)
        pca.fit(self.train[columns])
        self.train[new_col_name] = pca.transform(self.train[columns])
        self.test[new_col_name] = pca.transform(self.test[columns])
        self.train.drop(columns=columns, inplace=True)
        self.test.drop(columns=columns, inplace=True)

    def __process_rainfall(self, prefixes):
        for month in range(1, 13):
            combination_cols = []
            for year in range(2008, 2019):
                suffix = f"{'0' if month < 10 else ''}{month}_{year}"
                for name in prefixes:
                    combination_cols.append(f"{name}{suffix}")

            column_name = f"pca_{'BioTas' if len(prefixes) == 3 else 'BioPr'}_"
            column_name = f"{column_name}{'0' if month < 10 else ''}{month}"
            self.__apply_pca(combination_cols, column_name)

    def __balance_test_cols(self):
        yy_cols = [col for col in self.train if "speciesId" in col]
        true_cols = self.train.drop(columns=yy_cols).columns
        new_cols = list(set(true_cols) - set(self.test.columns))
        self.test[new_cols] = False 
        self.test = self.test[true_cols]

    def __final_drop(self):
        for prefix in (self.__tas_prefixes + self.__pr_prefix):
            drop_cols = [col for col in self.train if prefix in col]
            self.train.drop(columns=drop_cols, inplace=True)
            self.test.drop(columns=drop_cols, inplace=True)





