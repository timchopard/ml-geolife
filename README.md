# GeoLifeCLEF 2024


1. [Data](#data)  
1.1 [Data Loader](#data-loader)
2. [Structure](#structure)  

## Data
[kaggle page](https://www.kaggle.com/competitions/geolifeclef-2024)

### Data Loader
The data loader is comprised of three classes  ```CSVLoader```, ```ImageLoader``` and ```DataLoader```, each stored in their respective files as seen in the [file structure](#structure).

The first two load the training and test CSV files and their associated images. ```DataLoader``` inherits from both classes to feed the data directly to a model. 

## Structure
```bash
.
├── datamanagement
│   ├── csvloader.py
│   ├── dataloader.py
│   └──imageloader.py
├── README.md
├── scratch.ipynb
├── scratchpad.ipynb
└── scratch.py
```