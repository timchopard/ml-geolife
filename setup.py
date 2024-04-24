#!/usr/bin/env python
"""Downloads and preprocesses the kaggle data
"""

from pipeline.getdata import DirectoryManagement
from pipeline.preprocess import DataCollector

DirectoryManagement()
DataCollector().process_and_save_data()
