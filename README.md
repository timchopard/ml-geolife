# GeoLifeCLEF 2024

This repository contains the models used for the 2024 GeoLifeClef competition as the team BernIgen (an exceedingly clever name derived from the cities of Bern and Groningen).
We (ranked 5th in this competition)[https://www.kaggle.com/competitions/geolifeclef-2024/leaderboard].

## Contents
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

#### `Preprocessor` Class

The preprocessor class found in `./preprocessor.py` loads in the CSV data for the Presence/Absence surveys and performs the following operations:

- Drops unnecessary columns (default set as `['geoUncertaintyInM', 'country']`)
- _training data only:_ Removes the less common species from the data (default set as top 500 species kept)
- Generates the elevation data from the longtitude and latitude values
- _training data only:_ One-hot encodes the species data
- One-hot encodes the region data

TODO: Currently this leaves the test data with too few columns for some models, due to it containing fewer regions than the training data. This could be fixed.

#### `ImagePreprocessor` Class

The image preprocessor is also found in `./preprocessor.py`. This class loads in the raw images and combines them into four channel (Red, Green, Blue, Infrared) images stored as numpy arrays.

### Postprocessor

The postprocessor takes the generated outputs of a model and converts it into the format used for submissions.

- `process` Can be called to return a dataframe
- `save` Can instead be called to save the data to csv at a specified path
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

| **Ref** | **Model**                                                                                                           | **Kaggle Score** | **Date**   |
| ------- | ------------------------------------------------------------------------------------------------------------------- | ---------------- | ---------- |
| 1       | positive_weigh_factor 0.9 (no sentinal)                                                                             | 0.33180          | 2024-05-13 |
| 2       | 2x resnet18 (no sentinal, seed 113) : 1x XGBoost                                                                    | 0.33272          | 2024-05-13 |
| 3       | Updated xgb hyperparameters                                                                                         | 0.34329          | 2024-05-15 |
| 4       | Original XGB and VIT, +5                                                                                            | 0.34224          | 2024-05-15 |
| 5       | As previous but with +5 to counts                                                                                   | 0.33605          | 2024-05-15 |
| 6       | XGB with hyperparameters tuned and training only on species with more than 200 occurrences in the engineered data   | 0.33253          | 2024-05-15 |
| 7       | 1x pre-trained ViT : 3x resnet18 : 1x XGBoost                                                                       | 0.33946          | 2024-05-14 |
| 8       | 1x our trained ViT : 2x resnet18 : 1x XGBoost                                                                       | 0.33504          | 2024-05-14 |
| 9       | 2x pre-trained ViT : 2x resnet18 : 1x XGBoost                                                                       | 0.34001          | 2024-05-13 |
| 10      | 1x pre-trained ViT : 2x resnet18 : 1x XGBoost                                                                       | 0.34132          | 2024-05-13 |
| 11      | 3:1 50-75nn:xgb                                                                                                     | 0.30289          | 2024-05-13 |
| 12      | 2x resnet18 (improved normalisation for sentinal images) : 1x XGBoost                                               | 0.33471          | 2024-05-13 |
| 13      | 3:2 nn50-30:xgb                                                                                                     | 0.32980          | 2024-05-12 |
| 14      | 3:1 nn50-30:xgb                                                                                                     | 0.33042          | 2024-05-12 |
| 15      | 5:1 nn50-30:xgb                                                                                                     | 0.32619          | 2024-05-12 |
| 16      | 2x resnet18 (no sentinal) : 1x XGBoost                                                                              | 0.33345          | 2024-05-11 |
| 17      | pure resnet50 - 30 epochs                                                                                           | 0.30198          | 2024-05-10 |
| 18      | 2x resnet18 : 1x XGBoost                                                                                            | 0.33350          | 2024-05-10 |
| 19      | AS before 2:1 weighting nn:xgb                                                                                      | 0.33103          | 2024-05-08 |
| 20      | 50-30, as before                                                                                                    | 0.32578          | 2024-05-08 |
| 21      | As before with 34, 20 epochs                                                                                        | 0.32355          | 2024-05-07 |
| 22      | As before, but using resnet34 instead of resnet18                                                                   | 0.28453          | 2024-05-07 |
| 23      | NN + previous (1:1)                                                                                                 | 0.32251          | 2024-05-07 |
| 24      | Ensemble Weighted counts (<0.15:1> country to xgb count model) Weighted predictions (<1.25:1> core data to landsat) | 0.28861          | 2024-05-07 |
| 25      | top 20 landsat + xgb + pca                                                                                          | 0.28269          | 2024-05-06 |
| 26      | xgb dynamic 7                                                                                                       | 0.23881          | 2024-05-05 |
| 27      | xgb dynamic 5                                                                                                       | 0.23855          | 2024-05-05 |
| 28      | uploadable_2024-05-03_16-47                                                                                         | 0.23855          | 2024-05-03 |
| 29      | from 2010, dynamic, +5                                                                                              | 0.17199          | 2024-05-03 |
| 30      | Full dataset, dynamic + 5                                                                                           | 0.25071          | 2024-05-02 |
| 31      | uploadable_2024-05-02_16-10                                                                                         | 0.23168          | 2024-05-02 |
| 32      | dynamic n, xgboost                                                                                                  | 0.23517          | 2024-05-02 |
| 33      | uploadable_2024-05-01_16-45                                                                                         | 0.23851          | 2024-05-01 |
| 34      | uploadable_2024-05-01_16-44                                                                                         | 0.24070          | 2024-05-01 |
| 35      | uploadable_2024-05-01_16-40                                                                                         | 0.24019          | 2024-05-01 |
| 36      | uploadable_EnsembleXGB                                                                                              | 0.11286          | 2024-04-24 |
| 37      | nearest neighbour                                                                                                   | 0.19451          | 2024-04-22 |
| 38      | tree                                                                                                                | 0.12993          | 2024-04-19 |
| 39      | Basic Decision Tree - Top 500                                                                                       | 0.18229          | 2024-04-11 |

#### Comparisions

##### Positive Weight Factor

Tested on 2x resnet18 (no sentinal) : 1x XGBoost

| **Positive Weight Factor** | **Kaggle Score** |
| -------------------------- | ---------------- |
| 0.9                        | 0.33180          |
| 1.0                        | 0.33345          |
| 1.1                        | 0.?              |
