#!/usr/bin/env python
"""Downloads and preprocesses the kaggle data
"""
import sys
from pipeline.getdata import DirectoryManagement

# from pipeline.preprocess import pickle_landsat
from pipeline.preprocess import PCACreator
from pipeline.preprocess import DataCollector

if __name__ == "__main__":
    args = sys.argv
    if "-h" in args or len(args) == 1:
        print(
            ":: Flags:\n\t-d  :  Download the data from kaggle (requires "
            + "kaggle CLI)\n\t-p  :  Preprocess and pickle the data as single "
            + "train and test files\n\t-r  :  Reduce the dimensions of the "
            + "data using PCA, saving the PCA models for later use"
        )
    if "-d" in args:
        DirectoryManagement()
    if "-p" in args:
        DataCollector().process_and_save_data()
    if "-r" in args:
        PCACreator(method="full")
    if "-l" in args:
        pickle_landsat()
