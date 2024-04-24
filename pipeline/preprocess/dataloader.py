""" 
from preprocess.dataloader import load_x_y
from preprocess.dataloader import load_with_split
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _handle_drops(data, drop_type):
    """Helper function.
    Handles the "drop" key word argument
    """
    if "soil" in drop_type or "def" in drop_type:
        data = data.drop(columns=[
            "Soilgrid-bdod",
            "Soilgrid-cec",
            "Soilgrid-cfvo",
            "Soilgrid-clay",
            "Soilgrid-nitrogen",
            "Soilgrid-phh2o",
            "Soilgrid-sand",
            "Soilgrid-silt",
            "Soilgrid-soc",
        ])
    if "small" in drop_type or "def" in drop_type: 
        data = data.dropna(subset=[
            "Elevation",
        ])
    if "safe" in drop_type or "def" in drop_type:
        data["geoUncertaintyInM"] = data["geoUncertaintyInM"].replace(
            np.nan, 
            data["geoUncertaintyInM"].median(),
        )
    if "human" in drop_type or "def" in drop_type:
        data = data.drop(columns=[
            "HumanFootprint-NavWater1994",
            "HumanFootprint-NavWater2009",
            "HumanFootprint-Roads",
            "HumanFootprint-HFP1993",
            "HumanFootprint-HFP2009",
        ])
    return data

def _handle_area(data, area_type):
    """Helper function.
    Handles the "area" key word argument
    """
    if area_type == "dropcol":
        return data.drop(columns="areaInM2")
    if area_type == "droprow":
        return data.dropna(subset=["areaInM2"])
    if area_type == "median":
        data["areaInM2"] = data["areaInM2"].fillna(data["areaInM2"].median())
        return data
    if area_type == "mean":
        data["areaInM2"] = data["areaInM2"].fillna(data["areaInM2"].mean())
        return data
    return data

def _handle_filtering(data, filter_method):
    if not filter_method:
        return data 
    filter_mask = [col for col in data if col.startswith('Bio-') and           \
                   '06_2018' not in col and '12_2018' not in col]
    return data.drop(columns=filter_mask)

def load_x_y(is_train: bool, **kwargs):
    """Loads in the pickled data and splits it into features and labels 
    dataframes

    args:
        is_train        :   True to load in train, False for test
    kwargs:
        drop            :   How to drop not a number values:
                                "soil"  Drop soil columns with NaN 
                                "small" Drop rows from columns with few NaN
                                "human" Drop human footprint columns with NaN
                                "def"   All of the above
        area            :   How to handle the Area in M2 columns (~12000 NaN)
                                "dropcol"   Drop the column
                                "droprow"   Drop the rows with NaN
                                "median"    Replace NaN with median values
                                "mean"      Replace NaN with mean values
        filtering       :   True to filter out a large number of columns
    return:
        pd.DataFrame        Features
        pd.DataFrame        Labels
    """
    data = pd.read_pickle(
        f"data/processed/{'train' if is_train else 'test'}_full.pkl"
    )
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    if "drop" in kwargs:
        data = _handle_drops(data, kwargs.pop("drop"))
    if "area" in kwargs:
        data = _handle_area(data, kwargs.pop("area"))
    if "filtering" in kwargs:
        data = _handle_filtering(data, kwargs.pop("filtering"))
    yy_filter = [col for col in data if col.startswith('speciesId')]
    if is_train:
        return data.drop(columns=yy_filter), data[yy_filter]
    return data

def load_with_split(test_size: float = 0.2, seed: int = None, **kwargs):
    """Loads in the data using the load_x_y function and splits it into 
    train and test. 
    Passes kwargs directly to load_x_y

    args:
        test_size       :   The proportion of the data to allocate to test
        seed            :   The seed for the train_test_split random state
    kwargs:
        *               :   As in load_x_y
    """
    xx, yy = load_x_y(is_train=True, **kwargs)
    return train_test_split(xx, yy, test_size=test_size, random_state=seed)

def match_test_columns(test_data, model_columns):
    missing = list(set(model_columns) - set(test_data.columns)) 
    test_data[missing] = False 
    return test_data[model_columns]