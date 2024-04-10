# GeoLifeCLEF 2024

1. [Data](#data)
   1. [Data Loader](#data-loader)
   1. [Images as numpy](#images-as-numpy-arrays)
1. [Structure](#structure)
1. [Potential Experiment](#potential-experiments)
   1. [Data](#data)
   1. [Models](#models)

## Data

[kaggle page](https://www.kaggle.com/competitions/geolifeclef-2024)

### Data Loader

The data loader is comprised of three classes `CSVLoader`, `ImageLoader` and `DataLoader`, each stored in their respective files as seen in the [file structure](#structure).

The first two load the training and test CSV files and their associated images. `DataLoader` inherits from both classes to feed the data directly to a model.

### Images as numpy arrays

Running `python imageprocessor.py -d` from the root will convert train and test images into four `.npz` files located in a new directory (`processed_images/`) unless otherwise specified.

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
