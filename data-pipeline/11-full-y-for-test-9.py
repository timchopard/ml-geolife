import pandas as pd

# Read in full species data and one hot encode it
yy = pd.read_csv("_raw/GLC24_PA_metadata_train.csv", index_col="surveyId")
yy = pd.get_dummies(yy["speciesId"].astype(int), prefix="species")

# Load in the survey ID values for the test data
test_surveys = list(pd.read_pickle("output/9-int-split-test.pkl")["surveyId"])

# Strip out non-test surveys and group by remaining survey IDs
yy = yy[yy.index.isin(test_surveys)]
yy = yy.groupby(by=yy.index).max()

# Pickle file
yy.to_pickle("output/9-int-split-test-full-labels.pkl")