""" from pipeline.getdata import DirectoryManagement
"""
import os
import subprocess

class DirectoryManagement():
    """ Ensures that the correct directory structure is in place before using
    a bash script to download the raw data from kaggle
    """

    dir_path = "data/"
    raw_path = "data/_raw/"
    processed_path = "data/processed/"

    def __init__(self, **kwargs):
        """ Interprets kwargs and calls relevant functions
        
        kwargs:
            dirs        :   False to not check and create directories
            download    :   False to not download data
        """
        if "dirs" not in kwargs or not kwargs.pop("dirs"):
            self.check_make_dirs()
        if "download" not in kwargs or not kwargs.pop("downloads"):
            self.download_data()

    def check_make_dirs(self):
        """ Checks if directories exist, and creates them if not.
        """
        print(":: Setting up dirs:\n\tdata/_raw/\n\t    /processed/")
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)
        if not os.path.exists(self.raw_path):
            os.mkdir(self.raw_path)
        if not os.path.exists(self.processed_path):
            os.mkdir(self.processed_path)

    def download_data(self):
        """ Runs a bash script to download data from kaggle
        """
        subprocess.run(['bash', 'pipeline/getdata/downloaddata.sh'])
