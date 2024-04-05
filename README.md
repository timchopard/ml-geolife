# GeoLifeCLEF 2024


1. [Data](#data)  
1.1 [Data Loader](#data-loader)  
1.2 [Images as numpy](#images-as-numpy-arrays)
2. [Structure](#structure)  

## Data
[kaggle page](https://www.kaggle.com/competitions/geolifeclef-2024)

### Data Loader
The data loader is comprised of three classes  ```CSVLoader```, ```ImageLoader``` and ```DataLoader```, each stored in their respective files as seen in the [file structure](#structure).

The first two load the training and test CSV files and their associated images. ```DataLoader``` inherits from both classes to feed the data directly to a model. 

### Images as numpy arrays

Running ```python imageprocessor.py -d``` from the root will convert train and test images into four ```.npz``` files located in a new directory (```processed_images/```) unless otherwise specified.

## Structure
```bash
.
├── processed_images
│   └── ... [Local only]
├── data
│   └── ... [Local only]
├── datamanagement
│   ├── csvloader.py
│   ├── dataloader.py
│   └── imageloader.py
├── README.md
├── scratch.ipynb
├── scratchpad.ipynb
└── scratch.py
```