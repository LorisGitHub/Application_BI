#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer

# Turn interactive plotting off
plt.ioff()

# read input text and put data inside a data frame
data = pd.read_csv('data/data_mining_DB_clients_tbl.csv')
data_bis = pd.read_csv('data/data_mining_DB_clients_tbl_bis.csv')

print(data.head())

# print nb of instances and features
print(data.shape)

# print feature types
print(data.dtypes)

data_feature_names = ['CDSEXE','MTREV','NBENF','CDSITFAM','DTADH','CDTMT','CDDEM','DTDEM',
                'ANNEE_DEM','CDMOTDEM','CDCATCL','AGEAD','rangagead','agedem','rangagedem',
                'rangdem','adh','rangadh']
x = data[data_feature_names]
y = data['Id']

# Replace first file rangadh col missing values with empty string
data.rangadh.replace('NULL', '', inplace=True)


data_bis_feature_names = ['CDSEXE','DTNAIS','MTREV','NBENF','CDSITFAM','DTADH','CDTMT','CDMOTDEM','CDCATCL','Bpadh','DTDEM']
x_bis = data_bis[data_bis_feature_names]

# Concat both dataframe and remap indexes
newDf = pd.concat([x, x_bis], ignore_index=True)

# Delete CDDEM col because it only exist in first file
del newDf["CDDEM"]

# Delete Bpadh col because it only exist in first file
del newDf["Bpadh"]

# Next step is to finish the merge properly by inserting corrects values for each columns

# Replace col ANNEE_DEM missings values with the correct value (TODO)
# For this one, you need to get the corresponding DTDEM value and extract the year
newDf["ANNEE_DEM"] = np.where(np.isnan(newDf["ANNEE_DEM"]), newDf["DTDEM"].str[0:4], newDf["ANNEE_DEM"])

# Replace col rangagead missings values with the correct value (TODO)
newDf['rangagead'].fillna("2  26-30", inplace = True)

# Replace col agedem missings values with the correct value (TODO)
newDf['agedem'].fillna("51", inplace = True)

# Replace col rangagedem missings values with the correct value (TODO)
newDf['rangagedem'].fillna("7  51-55", inplace = True)

# Replace col rangdem missings values with the correct value (TODO)
newDf['rangdem'].fillna("3  2001", inplace = True)

# Replace col adh missings values with the correct value (TODO)
newDf['adh'].fillna("21", inplace = True)

# Replace col rangadh missings values with the correct value (TODO)
newDf['rangadh'].fillna("5  20-24", inplace = True)

# Replace col DTNAIS missings values with the correct value (TODO)
newDf['DTNAIS'].fillna("1961-07-04", inplace = True)

# Replace col AGEAD missings values with the correct value (TODO)
# For this one, you need to get the values of the corresponding DTADH and DTNAIS value and substract them
newDf["AGEAD"] = np.where(np.isnan(newDf["AGEAD"]), (newDf["DTADH"].str[0:4]).astype('int64') - (newDf["DTNAIS"].str[0:4]).astype('int64'), newDf["AGEAD"])

# Generate the new CSV file
newDf.to_csv('my_csv.csv', index_label="Id")

