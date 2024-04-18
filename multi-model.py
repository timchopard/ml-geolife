import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from skmultilearn.adapt import MLkNN
from sklearn.metrics import hamming_loss, accuracy_score
import joblib

# # ## Export the data

# train_df = pd.read_pickle("processed_data/internal/train.pkl")
# test_df = pd.read_pickle("processed_data/internal/test_X.pkl")


# # drop columns begin speciesId_
# X = train_df.drop(
#     columns=train_df.columns[train_df.columns.str.startswith("speciesId_")]
# )
# X = X.drop(columns="surveyId")
# y = X = train_df.drop(
#     columns=train_df.columns[~train_df.columns.str.startswith("speciesId_")]
# )


# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=113)
# X_train.to_pickle("tmpX_train.pk")
# X_val.to_pickle("tmpX_val.pk")
# y_train.to_pickle("tmpy_train.pk")
# y_val.to_pickle("tmpy_val.pk")

# X_train = pd.read_pickle("tmpX_train.pk")
# y_train = pd.read_pickle("tmpy_train.pk")
X_test = pd.read_pickle("processed_data/internal/test_X.pkl")

# rfc = RandomForestClassifier(random_state=113, verbose=1)

# rfc.fit(X_train, y_train)

rfc = joblib.load("rfc.pkl")

rfc.predict(X_test)


# using Multi-label kNN classifier
# mlknn_classifier = MLkNN()
# mlknn_classifier.fit(np.array(X_train), np.array(y_train))
