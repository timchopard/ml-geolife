
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

top500_df = pd.read_csv("../processed_data/partially_processed/top500.csv")

top500_df = top500_df.drop(columns=['geoUncertaintyInM', 'country'])

with pd.option_context('use_inf_as_na', True):
    top500_df.areaInM2 = top500_df.areaInM2.fillna(top500_df.areaInM2.mean())

top500_df.speciesId = top500_df.speciesId.astype('uint16')

ohe = OneHotEncoder(dtype='bool')

transformed = ohe.fit_transform(top500_df[['region', 'speciesId']])

top500_df[np.concatenate((ohe.categories_[0], ohe.categories_[1]), axis=0)] = transformed.toarray()

top500_df_grouped = top500_df.groupby('surveyId', as_index=False).max().copy()

print(top500_df_grouped.shape)

top500_df_grouped.to_pickle("../processed_data/partially_processed/top500_df_elevation_onehot_grouped.pkl")


