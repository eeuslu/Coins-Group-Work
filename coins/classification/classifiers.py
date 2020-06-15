import pandas as pd
import numpy as np
from scipy import stats
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier

from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error 
from sklearn.metrics import accuracy_score

################################################################################################################################################################
# define classification models

def logisticRegression(input, target, PCA_Value=0.99):
    x = input.copy()
    y = target.copy()

    # split data into test and training set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)
    
    # normalize data
    st_scaler = StandardScaler()
    x_train_scaled = st_scaler.fit_transform(x_train)
    x_test_scaled = st_scaler.transform(x_test)

    #if wanted reduce the dimensionalty, but keep 'PCA_Value' of the variance
    if PCA_Value < 1.0 and PCA_Value > 0.0:
        pca = PCA(PCA_Value)
        x_train_scaled = pca.fit_transform(x_train_scaled)
        x_test_scaled = pca.transform(x_test_scaled)

    # create regression
    log = LogisticRegression(n_jobs=-1)
    try:
        log.fit(x_train_scaled, y_train)
    except ValueError:
        print(y_train)
    # predict test set
    y_predict = log.predict(x_test_scaled)

    return (r2_score(y_test, y_predict)), (accuracy_score(y_test, y_predict)), log , pca, st_scaler



def gaussianNBClassifier(input, target, PCA_Value=0.99):
    x = input.copy()
    y = target.copy()

    # split data into test and training set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)
    
    # normalize data
    st_scaler = StandardScaler()
    x_train_scaled = st_scaler.fit_transform(x_train)
    x_test_scaled = st_scaler.transform(x_test)

    #if wanted reduce the dimensionalty, but keep 'PCA_Value' of the variance
    if PCA_Value < 1.0 and PCA_Value > 0.0:
        pca = PCA(PCA_Value)
        x_train_scaled = pca.fit_transform(x_train_scaled)
        x_test_scaled = pca.transform(x_test_scaled)

    # create regression
    gnb = GaussianNB()
    gnb.fit(x_train_scaled, y_train)

    # predict test set
    y_predict = gnb.predict(x_test_scaled)

    return (r2_score(y_test, y_predict)), (accuracy_score(y_test, y_predict)), gnb, pca, st_scaler



def ridgeClassifier(input, target, PCA_Value=0.99):
    x = input.copy()
    y = target.copy()

    # split data into test and training set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)
    
    # normalize data
    st_scaler = StandardScaler()
    x_train_scaled = st_scaler.fit_transform(x_train)
    x_test_scaled = st_scaler.transform(x_test)

    #if wanted reduce the dimensionalty, but keep 'PCA_Value' of the variance
    if PCA_Value < 1.0 and PCA_Value > 0.0:
        pca = PCA(PCA_Value)
        x_train_scaled = pca.fit_transform(x_train_scaled)
        x_test_scaled = pca.transform(x_test_scaled)

    # create regression
    rdg = RidgeClassifier()
    rdg.fit(x_train_scaled, y_train)

    # predict test set
    y_predict = rdg.predict(x_test_scaled)

    return (r2_score(y_test, y_predict)), (accuracy_score(y_test, y_predict)), rdg, pca, st_scaler



def randomForestClassifier(input, target, PCA_Value=0.99):
    x = input.copy()
    y = target.copy()

    # split data into test and training set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)
    
    # normalize data
    st_scaler = StandardScaler()
    x_train_scaled = st_scaler.fit_transform(x_train)
    x_test_scaled = st_scaler.transform(x_test)

    #if wanted reduce the dimensionalty, but keep 'PCA_Value' of the variance
    if PCA_Value < 1.0 and PCA_Value > 0.0:
        pca = PCA(PCA_Value)
        x_train_scaled = pca.fit_transform(x_train_scaled)
        x_test_scaled = pca.transform(x_test_scaled)

    # create regression
    rfc = RandomForestClassifier(n_jobs=-1)
    rfc.fit(x_train_scaled, y_train)

    # predict test set
    y_predict = rfc.predict(x_test_scaled)

    return (r2_score(y_test, y_predict)), (accuracy_score(y_test, y_predict)), rfc , pca, st_scaler



def knnClassifier(input, target, k, PCA_Value=0.99):
    x = input.copy()
    y = target.copy()

    # split data into test and training set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)
    
    # normalize data
    st_scaler = StandardScaler()
    x_train_scaled = st_scaler.fit_transform(x_train)
    x_test_scaled = st_scaler.transform(x_test)

    #if wanted reduce the dimensionalty, but keep 'PCA_Value' of the variance
    if PCA_Value < 1.0 and PCA_Value > 0.0:
        pca = PCA(PCA_Value)
        x_train_scaled = pca.fit_transform(x_train_scaled)
        x_test_scaled = pca.transform(x_test_scaled)

    # create regression
    knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=k)
    knn.fit(x_train_scaled, y_train)

    # predict test set
    y_predict = knn.predict(x_test_scaled)

    return (r2_score(y_test, y_predict)), (accuracy_score(y_test, y_predict)), knn, pca, st_scaler



def svcLinear(input, target, PCA_Value=0.99):
    x = input.copy()
    y = target.copy()

    # split data into test and training set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)
    
    # normalize data
    st_scaler = StandardScaler()
    x_train_scaled = st_scaler.fit_transform(x_train)
    x_test_scaled = st_scaler.transform(x_test)

    #if wanted reduce the dimensionalty, but keep 'PCA_Value' of the variance
    if PCA_Value < 1.0 and PCA_Value > 0.0:
        pca = PCA(PCA_Value)
        x_train_scaled = pca.fit_transform(x_train_scaled)
        x_test_scaled = pca.transform(x_test_scaled)

    # create regression
    svc = SVC(kernel="linear", max_iter=50)
    svc.fit(x_train_scaled, y_train)

    # predict test set
    y_predict = svc.predict(x_test_scaled)

    return (r2_score(y_test, y_predict)), (accuracy_score(y_test, y_predict)), svc , pca, st_scaler



def svcPoly(input, target, d, PCA_Value=0.99):
    x = input.copy()
    y = target.copy()

    # split data into test and training set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)
    
    # normalize data
    st_scaler = StandardScaler()
    x_train_scaled = st_scaler.fit_transform(x_train)
    x_test_scaled = st_scaler.transform(x_test)

    #if wanted reduce the dimensionalty, but keep 'PCA_Value' of the variance
    if PCA_Value < 1.0 and PCA_Value > 0.0:
        pca = PCA(PCA_Value)
        x_train_scaled = pca.fit_transform(x_train_scaled)
        x_test_scaled = pca.transform(x_test_scaled)

    # create regression
    svc = SVC(kernel="poly", degree=d, max_iter=50)
    svc.fit(x_train_scaled, y_train)

    # predict test set
    y_predict = svc.predict(x_test_scaled)

    return (r2_score(y_test, y_predict)), (accuracy_score(y_test, y_predict)), svc , pca, st_scaler

