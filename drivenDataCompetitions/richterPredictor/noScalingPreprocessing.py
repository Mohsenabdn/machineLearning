# Preprocessing features without scaling
###############################################################

import pandas as pd

# Reading data
df = pd.read_csv('train_values.csv', index_col='building_id')

# Categorical to numeric
dfCat = df.select_dtypes(include=['object'])
dfCatToNum = pd.get_dummies(dfCat, prefix=dfCat.columns)

# Dropping categorical columns
df = df.drop(dfCat.columns, axis=1)

# Concatenating numeric columns
dfPP = pd.concat([df, dfCatToNum], axis=1)

# Producing the csv file
dfPP.to_csv('noScalingPPTrainValues.csv')