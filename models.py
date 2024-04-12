import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


class ModelGeneric():

    def __init__(self, 
                 data_path:str = "processed_data/top_500_species_train.pkl"):
        """
        """
        if data_path[-3:] == "pkl":
            data = pd.read_pickle(data_path)
        if data_path[-3:] == "csv":
            data = pd.read_csv(data_path)

        self.XX, self.YY = data[data.columns[:13]], data[data.columns[13:]]

    def split_data(self, split_point:float = 0.2, remove_original:bool = True,
                   seed:int = 31):
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.XX, self.YY, test_size=split_point, random_state=seed
        )
        if remove_original:
            self.XX = None 
            self.YY = None 

    def get_f1(self):
        predictions = self.model.predict(self.X_val)
        score = f1_score(self.Y_val.to_numpy(), predictions, average='micro')
        print(f":: [F1] {score}")


class DecisionTreeModel(ModelGeneric):
    from sklearn.tree import DecisionTreeRegressor

    def __init__(self, 
                 data_path:str = "processed_data/top_500_species_train.pkl"):
        super().__init__(data_path)

    def train(self):
        """
        """
        self.split_data()
        self.model = self.DecisionTreeRegressor()
        self.model.fit(self.X_train, self.Y_train)

    def save(self, dir="models", filepath="decision_tree"):
        """
        """
        print(":: [TODO] Fix and implement this method")
        # if not os.path.exists(dir):
        #     os.makedirs(dir)
        # if not os.path.isfile(filepath):
        #     open(f"{dir}/{filepath}", "x")
        # pickle.dump(self.model, open(f"{dir}/{filepath}", "wb"))
        # print(f":: [DecisionTreeRegressor] saved to \'{dir}/{filepath}\'")

class XGBoostModel(ModelGeneric):
    from xgboost import XGBRegressor
    from sklearn.model_selection import RepeatedKFold
    
    def __init__(self, 
                 data_path:str = "processed_data/top_500_species_train.pkl"):
        super().__init__(data_path)

    def train(self):
        """
        """
        self.split_data()
        self.model = self.XGBRegressor()
        evaluation = self.RepeatedKFold(
            n_splits=10, n_repeats=3, random_state=31)
        self.model.fit(self.X_train, self.Y_train)



