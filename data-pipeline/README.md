# Data Pipeline

A series of files to tidy the data. They are designed to be run in series, using the indicated numbers, but can be run individually. To run in series run `./run_me.sh`. When run individually the behaviours of the files can be modified by command-line arguments. These can be seen by running `python FILE --help` e.g. `python 1-top-n.py --help`. The final files are:

- `train_X.pkl`
- `train_y.pkl`
- `train_y_count.pkl` this is the number of species found in each survey
- `test_X.pkl`
- `test_y.pkl`
- `test_y_count.pkl` this is the number of species found in each survey
- `submission_actual.csv` this is `test_y.pkl` formatted as a submission file

## Processing Files

1. `1-top-n.py` restricts the data to the top _n_ species (default: 500)
1. `2-img-to-png.py` converts the 3-channel RGB and 1-channel NIR jpegs in to 4-channel pngs
1. `3-metadata-tidy.py` removes countries and regions not in the test data
1. `4-onehotencode.py` one hot encodes country, region, and species data
1. `5-groupby.py` groups the metadata file by surveyId
1. `6-combine-test.py` merges the additional datafiles (not timeseries) by surveyId
1. `7-combine-train.py` merges the additional datafiles (not timeseries) by surveyId
1. `8-tidy.py` drops rows with inf or NaN data
1. `9-data-internal.py` - splits the training data file into an internal train and test files with an 80:20 split
1. `10-finalise.py` - drops surveyId, and splits of the species data, creating train_X.pkl, train_y.pkl, train_y_count.pkl, test_X.pkl, test_y.pkl, test_y_count.pkl

## Folders

- `output` the output data including intermediate stages
- `_raw` the original data files from Kaggle

## Utilities

- `demo.ipynb` demo file showing the data being classified using a Random Forest classifier.
- `run_me.sh` bash script to run all the python files in order (note: you may need to run `chmod +x run_me.sh` to make this executable).

## To Do

- [ ] Process timeseries data
- [ ] Turn the image data in to pickple files containing all the data
- [ ] Potentially drop the outliers
- [ ] Create full training data
