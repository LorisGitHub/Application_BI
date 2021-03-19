#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from R_square_clustering import r_square
from scipy.cluster.hierarchy import dendrogram, linkage
np.set_printoptions(threshold=np.inf)

#provient du tp de data mining
#correlation_circle
def correlation_circle(df,nb_var,x_axis,y_axis):
    fig, axes = plt.subplots(figsize=(8,8))
    axes.set_xlim(-1,1)
    axes.set_ylim(-1,1)
    # label with variable names
    for j in range(nb_var):
        # ignore two first columns of df: Nom and Code^Z
        plt.annotate(df.columns[j],(corvar[j,x_axis],corvar[j,y_axis]))
    # axes
    plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
    # add a circle
    cercle = plt.Circle((0,0),1,color='blue',fill=False)
    axes.add_artist(cercle)
    plt.savefig('fig/acp_correlation_circle_axes_'+str(x_axis)+'_'+str(y_axis))
    plt.close(fig)

def readDataFromCSV(path, verbose):
    data = pd.read_csv(path)
    if verbose:
        print(data.head())
        print(data.shape)
        print(data.dtypes)
    return data


def splitData(data):
    feature_names_raw = data.columns[1:19]
    X = data[feature_names_raw]
    Y = data['Id']
    C = data['demissionaire']
    return (X, Y, C)


def replaceMissingValues(data):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=np.nan)
    return pd.DataFrame(imp_mean.fit_transform(data), columns=data.columns)


def showAbberantValues(data):
    indexNames = data[data["MTREV"] > 20000].index
    print(indexNames)


def disjonctDataColumn(data, colName, delete):
    for item in data[colName].unique():
        data[item] = np.where(data[colName] == item, 1, 0)
    if delete:
        del data[colName]


def delDataColumns(data, colList):
    for item in colList:
        del data[item]


def scaleDataColumns(data, colList):
    data = data.astype('int64')
    standardScaler = StandardScaler()
    for item in colList:
        data[item] = pd.DataFrame(standardScaler.fit_transform(data[item].values.reshape(-1,1)))
    return data


def doAcp(data, components_number):
    acp = PCA(svd_solver='full', n_components=components_number)
    dataACP = acp.fit_transform(data)

    print(f">> Number of components: {acp.n_components_}")
    print(f">> Explained variance scores: {acp.explained_variance_ratio_}")
    print(f">> Singular values: {acp.singular_values_}")

    pca_df = pd.DataFrame(dataACP)

    for i in range(0, components_number, 2):
        plt.scatter(pca_df[i], pca_df[i+1], alpha=0.2, color="black")
        plt.xlabel('PCA '+str(i+1))
        plt.ylabel('PCA '+str(i+2))
        plt.savefig('fig/ACP_'+str(i+2)+'PC')
        plt.close()

    features = range(acp.n_components_)
    plt.bar(features, acp.explained_variance_ratio_, color='black')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(features)
    plt.savefig('fig/cp_variances')
    plt.close()

    pcaColumns = []
    for i in range (1, components_number+1):
        pcaColumns.append("PC"+str(i))

    loadings = acp.components_.T * np.sqrt(acp.explained_variance_)
    loading_matrix = pd.DataFrame(loadings, columns=pcaColumns, index=data.columns)
    print(loading_matrix)

    return pca_df


def elbowMethod(data):
    lst_k = range(2, 20)
    lst_rsq = []
    for k in lst_k:
        est = KMeans(n_clusters=k)
        est.fit(data)
        lst_rsq.append(r_square(data.to_numpy(), est.cluster_centers_,est.labels_,k))
    
    fig = plt.figure()
    plt.plot(lst_k, lst_rsq, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Variation du R2')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig('fig/k-means_elbow_method')
    plt.close()


def dendrogram(data):
    # Génère le dendrogram, prend plusieurs minutes à s'éxécuter, 
    linkage_matrix = linkage(data, 'ward')
    fig = plt.figure()
    dendrogram(
        linkage_matrix,
        color_threshold=0,
    )
    plt.title('Hierarchical Clustering Dendrogram (Ward)')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.tight_layout()
    plt.savefig('fig/hierarchical-clustering.svg')
    plt.close()
    

def main():
    plt.ioff()
    data = readDataFromCSV('data/fused_files.csv', True)
    (X, Y, C) = splitData(data)
    X = replaceMissingValues(X)
    showAbberantValues(X)
    disjonctDataColumn(X, "CDSITFAM", True)
    delDataColumns(X, ['DTADH', 'rangadh', 'rangagead', 'rangagedem', 'rangdem', 'agedem', 'DTDEM', 'ANNEE_DEM', 'CDMOTDEM', 'CDDEM'])
    scaledData = scaleDataColumns(X, ['MTREV', 'NBENF', 'CDSEXE', 'CDTMT', 'adh', 'AGEAD', 'CDCATCL'])
    pcaDf = doAcp(scaledData, 10)
    # elbowMethod(pcaDf)
    dendrogram(pcaDf)
    scaledData["demissionaire"] = C
    scaledData.to_csv('data/cleanedDataMining.csv', index_label="Id")


if __name__ == "__main__":
    main()