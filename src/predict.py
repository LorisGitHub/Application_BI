#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics


def accuracy_score(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        scores = cross_val_score(clf, X, y, cv=5)
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def confusion_matrix(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        predicted = cross_val_predict(clf, X, y, cv=5) 
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f" % metrics.accuracy_score(y, predicted))
        print(metrics.confusion_matrix(y, predicted))


def read_data_from_csv(path, verbose):
    data = pd.read_csv(path)
    if verbose:
        print(data.head())
        print(data.shape)
        print(data.dtypes)
        print(data["demissionaire"].value_counts())
    return data


def predict(X, Y, lst_classif, lst_classif_names):
    accuracy_score(lst_classif, lst_classif_names, X, Y)
    confusion_matrix(lst_classif, lst_classif_names, X, Y)


def decision_tree(X, Y, feature_names, dectree):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.2)
    dectree.fit(X_train, y_train)
    dectree.predict(X_test)

    fig = plt.figure(num=None, figsize=(8, 5), dpi=300)
    tree.plot_tree(dectree,  
                    feature_names=feature_names,  
                    class_names=["False","True"],  
                    filled=True, rounded=True)
    plt.savefig('fig/decision_tree')
    plt.close(fig)


def main():
    dummycl = DummyClassifier(strategy="most_frequent")
    gmb = GaussianNB()
    dectree = tree.DecisionTreeClassifier(max_depth = 3, random_state = 0)
    logreg = LogisticRegression(solver="liblinear")
    svc = svm.SVC(gamma='scale')

    lst_classif = [dummycl, gmb, dectree, logreg, svc]
    lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Logistic regression', 'SVM']

    data_train = read_data_from_csv('data/cleanedDataMining.csv', True)

    feature_names_raw = data_train.columns[1:20]
    X = data_train[feature_names_raw]
    Y = data_train['demissionaire']

    predict(X, Y, lst_classif, lst_classif_names)
    decision_tree(X, Y, feature_names_raw, dectree)


if __name__ == "__main__":
    main()



