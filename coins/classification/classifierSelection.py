import pandas as pd
import numpy as np
from scipy import stats
import itertools
from coins.correlation import calculateCorrWithPValue, balanceAccordingToColumn
from coins.io.output import saveBestResults, saveModel
from coins.io.input import loadBestResult, loadModel

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

from . import classifiers


################################################################################################################################################################
# find best classification model
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def findBestClassifier(x, y, targetDataFrameName, inputFeatureCombination=False, printProgress=False):
    #Used for classifies
    testSize = 0.3

    #initialize data
    arrayX= x.copy()
    dfY = y.copy()
    # create input feature
    targetFeature = dfY
    if len(arrayX) > 1:
        inputFeature = arrayX[0]
        i = 1
        while i < len(arrayX):
            inputFeature = pd.merge(inputFeature, arrayX[i], on='user_id', how='inner')
            i += 1
    else:
        inputFeature = arrayX[0]

    # calculate p-values
    inputFeature, targetFeature, dfJoinedData, pValues, _ = calculateCorrWithPValue(inputFeature, targetFeature)
  
    pValuesTransformed = pValues.T

    dfBestResults = pd.DataFrame(columns=['TargetFeature', 'InputFeature', 'BestAlgorithm', 'R^2', 'Accuracy', 'TestSampleSizePerValue'])
    targetFeatureList = list(pValues.index.values)
    
    # iterate through all target feature
    for targetFeature in targetFeatureList:
        # balance data based on target feature
        dfXYBalanced = balanceAccordingToColumn(dfJoinedData,targetFeature)
        testSampleSize = round(len(dfXYBalanced[dfXYBalanced[targetFeature]==dfXYBalanced[targetFeature].unique()[0]])*testSize,0)
  
        # get list with p_values bellow 0.05
        inputFeatureList = list((pValuesTransformed[pValuesTransformed[targetFeature] < 0.05]).index)
    
        #[targetFeature, inputFeatures, classifier, r2, accuracy, TestSampleSizePerValue]
        globalBestResult = [targetFeature, '-','-', -1, -1, testSampleSize]
        globalBestModel = '-'
        globalBestPCA = '-'
        globalBestScaler = '-'
    

        # check whether there are input features or not
        if len(inputFeatureList) > 0:
  
            # check whether a combination of input feature is wanted or not
            if inputFeatureCombination == True:
        
                # create all possible combinations of input features
                allInputFeatureCombinationsDummy = list(powerset(inputFeatureList))
                allInputFeatureCombinationsDummy.pop(0)

                allInputFeatureCombinations = []
                for inputFeatureCombination in allInputFeatureCombinationsDummy:
                    dummyList = []
                    for inputFeature in inputFeatureCombination:
                        dummyList.append(inputFeature)
                        
                    allInputFeatureCombinations.append(dummyList)
            else:
                # append all input Features in list
                allInputFeatureCombinations = []
                allInputFeatureCombinations.append(inputFeatureList)    
   
            # evaluate models for all combinations of input features
            for combination in allInputFeatureCombinations:

                x = dfXYBalanced[combination]
                y = dfXYBalanced[targetFeature]

                #create array for performance evaluation and initilize it with suitable values
                bestResult = [targetFeature, ('| '.join(combination)),'-', -1, -1, testSampleSize]
                bestModel = '-'
                bestPCA = '-'
                bestScaler = '-'

                # logistic regression
                try:
                    r2, accuracy, model, pca, scaler = classifiers.logisticRegression(x,y,testSize=testSize)
                except:
                    accuracy = -1
                if(accuracy > bestResult[4]):
                    bestResult[2] = 'Logistic Regression'
                    bestResult[3] = r2
                    bestResult[4] = accuracy
                    bestModel = model
                    bestPCA = pca
                    bestScaler = scaler

                # random forest classifier
                try:
                    r2, accuracy, model, pca, scaler = classifiers.randomForestClassifier(x,y,testSize=testSize)
                except:
                    accuracy = -1
                if(accuracy > bestResult[4]):
                    bestResult[2] = 'Random Forest Classifier'
                    bestResult[3] = r2
                    bestResult[4] = accuracy
                    bestModel = model
                    bestPCA = pca
                    bestScaler = scaler

                # K-Neighbors Classifier
                for k in range(1,10):
                    try:
                        r2, accuracy, model, pca, scaler = classifiers.knnClassifier(x,y,k,testSize=testSize)
                    except:
                        accuracy = -1
                    if(accuracy > bestResult[4]):
                        bestResult[2] = 'KNN Classifier, Degree: %d' % (k)
                        bestResult[3] = r2
                        bestResult[4] = accuracy
                        bestModel = model
                        bestPCA = pca
                        bestScaler = scaler

                # linear Support Vector Machine
                try:
                    r2, accuracy, model, pca, scaler = classifiers.svcLinear(x,y,testSize=testSize)
                except:
                    accuracy = -1
                if(accuracy > bestResult[4]):
                    bestResult[2] = 'SVC (linear)'
                    bestResult[3] = r2
                    bestResult[4] = accuracy
                    bestModel = model
                    bestPCA = pca
                    bestScaler = scaler

                # polynomial Support Vector Machine
                for d in range(1,10):
                    try:
                        r2, accuracy, model, pca, scaler = classifiers.svcPoly(x,y,d,testSize=testSize)
                    except:
                        accuracy = -1
                    if(accuracy > bestResult[4]):
                        bestResult[2] = 'SVC (polynomial), Degree: %d' % (d)
                        bestResult[3] = r2
                        bestResult[4] = accuracy
                        bestModel = model
                        bestPCA = pca
                        bestScaler = scaler

                # Gaussian Naive Bayes Classifier
                try:
                    r2, accuracy, model, pca, scaler = classifiers.gaussianNBClassifier(x,y,testSize=testSize)
                except:
                    accuracy = -1
                if(accuracy > bestResult[4]):
                    bestResult[2] = 'Gaussian Naive Bayes'
                    bestResult[3] = r2
                    bestResult[4] = accuracy
                    bestModel = model
                    bestPCA = pca
                    bestScaler = scaler

                # Ridge Regression
                try:
                    r2, accuracy, model, pca, scaler = classifiers.ridgeClassifier(x,y,testSize=testSize)
                except:
                    accuracy = -1
                if(accuracy > bestResult[4]):
                    bestResult[2] = 'Ridge Regression'
                    bestResult[3] = r2
                    bestResult[4] = accuracy
                    bestModel = model
                    bestPCA = pca
                    bestScaler = scaler

                # set bestResult to globalBestResult, if Accuracy is higher
                if bestResult[4] > globalBestResult[4]:
                    globalBestResult = bestResult
                    globalBestModel = bestModel
                    globalBestPCA = bestPCA
                    globalBestScaler = bestScaler

            # append best result to dfBestResult data frame
            dfBestResults.loc[len(dfBestResults)] = globalBestResult
            if(globalBestResult[4] > 0.0):
                saveModel(globalBestModel, globalBestPCA, globalBestScaler, targetFeature, targetDataFrameName)
                pass
        else:
            globalBestResult = [targetFeature, 'no input feature with p-value below 0.05' ,'-', '-', '-', '-']
            dfBestResults.loc[len(dfBestResults)] = globalBestResult
        if (printProgress == True):
            print("completed: " + targetFeature)
        saveBestResults(dfBestResults, targetDataFrameName)
    return dfBestResults

################################################################################################################################################################
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