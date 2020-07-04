import pandas as pd
import numpy as np
from scipy import stats
import itertools
from coins.io.input import loadBestResult, loadModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from . import classifiers

from coins.io.input import *
from coins.dfcreation.createPredictionDf import *


# predict new values
def predict(x1, x2, x3, x4, targetDataFrame): 
    # create input feature
    dfInputFeature = pd.merge(x1, x2, on='user_id', how='inner')
    dfInputFeature = pd.merge(dfInputFeature, x3, on='user_id', how='inner')
    dfInputFeature = pd.merge(dfInputFeature, x4, on='user_id', how='inner')

    dfInputFeature.reset_index(inplace=True)
    dfIndex = dfInputFeature["user_id"]
    dfInputFeature.drop(["user_id"], axis=1, inplace = True)

    # load bestResults csv
    try:
        dfBestResults = loadBestResult(targetDataFrame)
    except:
        return ("Fehler beim Einlesen der besten Resultate: Bitte trainiere zunächst ein Model für die Target Feature.")

    # go through all targetFeature and predict them
    for index, row in dfBestResults.iterrows():
        targetFeatureString = str(row["TargetFeature"])
        if row['BestAlgorithm'] != '-' and row['Accuracy'] != '0.0':
            inputFeatureList = row["InputFeature"].split("| ")
            # try to get all input feature
            try:
                inputFeature = dfInputFeature[inputFeatureList]
            except:
                return ("Fehler beim Extrahieren der InputFeature: Bitte übergebe alle notwendigen DataFrames.")
            # try to load model, pca and standardScaler
            try:
                print(targetFeatureString)
                model, pca, standardScaler = loadModel(targetFeatureString, targetDataFrame)
            except:
                return ("Fehler beim Laden der Modelle: Bitte trainiere zunächst ein Model für die Target Feature.")

            # predict new values
            x_scaled = standardScaler.transform(inputFeature)
            x_scaled = pca.transform(x_scaled)
            dfInputFeature[targetFeatureString] = model.predict(x_scaled)
    
    # concatenate with User ID
    dfInputFeature = pd.concat([dfIndex, dfInputFeature], axis=1)
    dfInputFeature.drop("index", axis=1, inplace=True)
    dfInputFeature.set_index("user_id", inplace=True)
    return dfInputFeature

# predict new values based on .csv files
def predictNewData(targetDataFrame):
    x = []
    if targetDataFrame != 'dfImageContents':
        try:
            # load .csv files
            images = loadDf('input/prediction/images_v2.csv')
            imageLabels = loadDf('input/imageLabels.csv')
        except:
            return("Bitte stelle alle benötigten csv-Dateien im Ordner input/prediction zur Verfügung. Ansonsten kann keine Prediction erstellt werden.")
        dfImageContents = preprocessImageContent(images, imageLabels)
        x.append(dfImageContents)
    
    if targetDataFrame != 'dfImageDescriptions':
        try:
            # load .csv files
            images = loadDf('input/prediction/images_v2.csv')
        except:
            return("Bitte stelle alle benötigten csv-Dateien im Ordner input/prediction zur Verfügung. Ansonsten kann keine Prediction erstellt werden.")
        dfImageDescriptions = preprocessImageDescriptions(images)
        x.append(dfImageDescriptions)
    
    if targetDataFrame != 'dfImageRating':
        try:
            # load .csv files
            images = loadDf('input/prediction/images_v2.csv')
        except:
            return("Bitte stelle alle benötigten csv-Dateien im Ordner input/prediction zur Verfügung. Ansonsten kann keine Prediction erstellt werden.")
        dfImageRatings = preprocessImageRatings(images)
        x.append(dfImageRatings)
    
    if targetDataFrame != 'dfPersonality':
        try:
            # load .csv files
            ipip = loadDf('input/prediction/ipip.csv')
        except:
            return("Bitte stelle alle benötigten csv-Dateien im Ordner input/prediction zur Verfügung. Ansonsten kann keine Prediction erstellt werden.")
        dfPersonality = preprocessPersonality(ipip)
        x.append(dfPersonality)

    if targetDataFrame != 'dfSocioDemographics':
        if targetDataFrame == 'dfPersonality':
            try:
                # load .csv files
                socioDemographics = loadDf('input/prediction/images_v2.csv')
            except:
                return("Bitte stelle alle benötigten csv-Dateien im Ordner input/prediction zur Verfügung. Ansonsten kann keine Prediction erstellt werden.")
        else:
            try:
                # load .csv files
                socioDemographics = loadDf('input/prediction/ipip.csv')
            except:
                return("Bitte stelle alle benötigten csv-Dateien im Ordner input/prediction zur Verfügung. Ansonsten kann keine Prediction erstellt werden.")

        dfSocioDemographics = preprocessSocioDemographics(socioDemographics)
        x.append(dfSocioDemographics)

    dfPrediction = predict(x[0], x[1], x[2], x[3], targetDataFrame)

    return dfPrediction