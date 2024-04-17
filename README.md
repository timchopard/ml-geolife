# GeoLifeCLEF 2024

1. [Data](#data)
   1. [Preprocessor](#preprocessor)
      1. [General Structure](#general-structure)  
      1. [Preprocessor Class](#preprocessor-class)
      1. [Image Preprocessor Class](#imagepreprocessor-class)
   1. [Postprocessor](#postprocessor)
   <!-- 1. [Images as numpy](#images-as-numpy-arrays) -->
1. [Structure](#structure)
1. [Potential Experiment](#potential-experiments)
   1. [Data](#data)
   1. [Models](#models)
1. [Submissions](#submissions)

## Data

[kaggle page](https://www.kaggle.com/competitions/geolifeclef-2024)

### General Structure

#### Country

Some of the countries have very few entries (~50 out of 1.5e6) and as such can be subsumed by adjacent countries

#### Region

The regions correlate to the map of Biogeographic regions below.

![Biogeographic Regions of Europe](regions.png "Biogeographic Regions of Europe. Source: https://en.wikipedia.org/wiki/Steppic_Biogeographic_Region#/media/File:Europe_biogeography_countries_en.svg")

Since some of these regions are also underrepresented in the data, some larger regions have absorbed others.

### Preprocessor  

#### ```Preprocessor``` Class
The preprocessor class found in ```./preprocessor.py``` loads in the CSV data for the Presence/Absence surveys and performs the following operations:

- Drops unnecessary columns (default set as ```['geoUncertaintyInM', 'country']```)  
- _training data only:_ Removes the less common species from the data (default set as top 500 species kept)  
- Generates the elevation data from the longtitude and latitude values  
- _training data only:_ One-hot encodes the species data 
- One-hot encodes the region data

TODO: Currently this leaves the test data with too few columns for some models, due to it containing fewer regions than the training data. This could be fixed.

#### ```ImagePreprocessor``` Class

The image preprocessor is also found in ```./preprocessor.py```. This class loads in the raw images and combines them into four channel (Red, Green, Blue, Infrared) images stored as numpy arrays.

### Postprocessor

The postprocessor takes the generated outputs of a model and converts it into the format used for submissions. 

- ```process``` Can be called to return a dataframe
- ```save``` Can instead be called to save the data to csv at a specified path
<!-- ### Data Loader -->
<!-- 
The data loader is comprised of three classes `CSVLoader`, `ImageLoader` and `DataLoader`, each stored in their respective files as seen in the [file structure](#structure).

The first two load the training and test CSV files and their associated images. `DataLoader` inherits from both classes to feed the data directly to a model. -->

<!-- ### Images as numpy arrays

Running `python imageprocessor.py -d` from the root will convert train and test images into four `.npz` files located in a new directory (`processed_images/`) unless otherwise specified. -->

## Structure

```
.
├── archive                      completed processes
├── processed_images             processed image data
│   └── ... [Local only]
├── processed_data               processed tabular data
│   └── ... [Local only]
├── data                         raw data
│   └── ... [Local only]
├── datamanagement               data manipulation library
│   ├── csvloader.py
│   ├── dataloader.py
│   └── imageloader.py
├── README.md
├── scratchpad-darren.ipynb
└── scratchpad-tim.ipynb
```

## Potential Experiments

### Data

- [ ] focus on top 100 species
- [ ] filter rare data
- [ ] balance data
- [ ] use bootstrapping

### Models

- [ ] seperate models for images and data
- [ ] generative adversarial networks (GAN)
- [ ] Vision Transformers (ViT)
- [ ] Contrastive Language–Image Pre-training (CLIP)

### Submissions

|   |   |   |
|---|---|---|
|__Model__|__Kaggle Score__|__Date__|
|Basic Decision Tree - Top 500| 0.18229 | 11.04.24|


