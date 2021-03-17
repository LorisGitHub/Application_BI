#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# initialize datetime
now = datetime.datetime.now()

# read input text and put data inside a data frame
data = pd.read_csv('data/data_mining_DB_clients_tbl.csv')
data_bis = pd.read_csv('data/data_mining_DB_clients_tbl_bis.csv')

feature_names_raw = data.columns[1:]
X = data[feature_names_raw]
Y = data['Id']

feature_names_raw_bis = data_bis.columns[1:]
X_bis = data_bis[feature_names_raw_bis]
Y_bis = data_bis['Id']

X_bis["AGEAD"] = (X_bis["DTADH"].str[0:4]).astype('int64') - (X_bis["DTNAIS"].str[0:4]).astype('int64')
X_bis["ANNEE_DEM"] = X_bis["DTDEM"].str[0:4]

# Concat both dataframe and remap indexes
newDf = pd.concat([X, X_bis], ignore_index=True)

# Map the entire adh column based on the current date if the client hasn't resigned or on his resignement date if he did 
newDf['adh'] = np.where((np.isnan(newDf['adh'])) & (newDf["ANNEE_DEM"] == '1900'), (now.year - (newDf["DTADH"].str[0:4]).astype('int64')), (newDf["ANNEE_DEM"].astype('int64') - (newDf["DTADH"].str[0:4]).astype('int64')))

# Drop rows with aberrant values
newDf = newDf[newDf["AGEAD"] < 100]
newDf = newDf[newDf["MTREV"] < 20000]
newDf = newDf.drop(newDf[(newDf["ANNEE_DEM"] == '1900') & (pd.isna(newDf['CDMOTDEM']) == False)].index)

# Create the target class
newDf['demissionaire'] = np.where(newDf["ANNEE_DEM"] == '1900', False, True)

del newDf["Bpadh"]
del newDf["DTNAIS"]

# Next step is to finish the merge properly by inserting mocking values for each columns
newDf['rangagead'].fillna("2  26-30", inplace = True)
newDf['agedem'].fillna("51", inplace = True)
newDf['rangagedem'].fillna("7  51-55", inplace = True)
newDf['rangdem'].fillna("3  2001", inplace = True)
newDf['rangadh'].fillna("5  20-24", inplace = True)

# Generate the new CSV file
newDf.to_csv('data/fused_files.csv', index_label="Id")

