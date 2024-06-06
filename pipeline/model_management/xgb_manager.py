import numpy as np
import pickle as pk

from xgboost import XGBRegressor 


class GeoLifeXGB():

    __main_params = {
        'device'        : 'cuda', 
        'reg_lambda'    : 12, 
        'learning_rate' : 0.1, 
        'min_split_loss': 0, 
        'reg_alpha'     : 0
    }

    __count_params = {
        'device'        : 'cuda',
        'learning_rate' : 0.1,
        'min_split_loss': 0,
        'reg_lambda'    : 10,
        'reg_alpha'     : 0, 
    }

    count_model = None
    main_model = None

    def __init__(self, seed) -> None:
        self.seed = seed 
        self.__load_data()

    def generate_count_model(self):
        self.count_model = XGBRegressor(**self.__count_params, seed=self.seed)
        self.count_model.fit(
            self.train.drop(columns=self.yy_cols),
            self.train[self.yy_cols].apply(lambda x: x.sum(), axis=1)
            
        )
        return self.count_model
    
    def predict_test_counts(self, addition:float = 3.5):
        if self.count_model is None:
            print("Train or load a model before predicting.")
            return 
        
        counts = self.count_model.predict(self.test) + addition 
        return counts.round().astype(int)

    def generate_main_model(self):
        self.main_model = XGBRegressor(**self.__main_params, seed=self.seed)
        combined_train = self.train.drop(columns=self.yy_cols)
        combined_train[
            ["landsat" + str(idx) for idx in range(360)]
        ] = self.landsat
        self.main_model.fit(combined_train, self.train[self.yy_cols])
        return self.main_model 
    
    def predict_test_scores(self):
        if self.main_model is None:
            print("Train or load a model before predicting.")
        combined_test = self.test 
        combined_test[
            ["landsat" + str(idx) for idx in range(360)]
        ] = self.landsat_test 
        return self.main_model.predict(combined_test)
 
    def __load_data(self):
        self.train = pk.load(open("data/processed/train.pkl", "rb"))
        self.test = pk.load(open("data/processed/test.pkl", "rb"))
        self.yy_cols = [col for col in self.train if "speciesId_" in col]
        self.landsat = pk.load(open("data/processed/train_ls.pkl", "rb"))
        self.landsat = self.landsat.reshape(self.landsat.shape[0], 360)
        self.landsat_test = pk.load(open("data/processed/test_ls.pkl", "rb"))
        self.landsat_test = self.landsat_test.reshape(
            self.landsat_test.shape[0], 
            360
        )
