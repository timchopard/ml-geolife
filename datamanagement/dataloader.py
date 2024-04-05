import pandas as pd 

from datamanagement.csvloader import CSVLoader
from datamanagement.imageloader import ImageLoader

class DataLoader(CSVLoader, ImageLoader):
    
    def __init__(self, train, test):
        super().__init__(train, test)