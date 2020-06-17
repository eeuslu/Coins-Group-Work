import pandas as pd
import numpy as np
from scipy import stats
from varname import nameof
import itertools
from coins.correlation import calculateCorrWithPValue, balanceAccordingToColumn
from coins.io.output import saveBestResults, saveModel

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


def findBestClassifier(x, y, targetDataFrameName ,inputFeatureCombination=False, printProgress=False):
    # initialize values
    arrayX = x.copy()
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
    dfBestResults = pd.DataFrame(columns=['TargetFeature', 'InputFeature', 'BestAlgorithm', 'R^2', 'Accuracy'])
    targetFeatureList = list(pValues.index.values)

    #print("target.f.l.  ",targetFeatureList)
    #print("p-T  ",pValuesTransformed)

    # iterate through all target feature
    for t in targetFeatureList:
        # balance data based on target feature
        dfXYBalanced = balanceAccordingToColumn(dfJoinedData,t)

        # get list with p_values bellow 0.05
        inputFeatureList = list((pValuesTransformed[pValuesTransformed[t] < 0.05]).index)
        globalBestResult = [t, '-','-', -1, -1]
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
                for a in allInputFeatureCombinationsDummy:
                    dummyList = []
                    for b in a:
                        dummyList.append(b)
                    allInputFeatureCombinations.append(dummyList)
            else:
                # append all input Features in list
                allInputFeatureCombinations = []
                allInputFeatureCombinations.append(inputFeatureList)        

            # evaluate models for all combinations of input features
            for combination in allInputFeatureCombinations:

                x = dfXYBalanced[combination]
                y = dfXYBalanced[t]

                #create array for performance evaluation and initilize it with suitable values
                bestResult = [t, ('| '.join(combination)),'-', -1, -1]
                bestModel = '-'
                bestPCA = '-'
                bestScaler = '-'

                # logistic regression
                r2, accuracy, model, pca, scaler = classifiers.logisticRegression(x,y)
                if(accuracy > bestResult[4]):
                    bestResult[2] = 'Logistic Regression'
                    bestResult[3] = r2
                    bestResult[4] = accuracy
                    bestModel = model
                    bestPCA = pca
                    bestScaler = scaler

                # random forest classifier
                r2, accuracy, model, pca, scaler = classifiers.randomForestClassifier(x,y)
                if(accuracy > bestResult[4]):
                    bestResult[2] = 'Random Forest Classifier'
                    bestResult[3] = r2
                    bestResult[4] = accuracy
                    bestModel = model
                    bestPCA = pca
                    bestScaler = scaler

                # K-Neighbors Classifier
                for k in range(1,10):
                    r2, accuracy, model, pca, scaler = classifiers.knnClassifier(x,y,k)
                    if(accuracy > bestResult[4]):
                        bestResult[2] = 'KNN Classifier, Degree: %d' % (k)
                        bestResult[3] = r2
                        bestResult[4] = accuracy
                        bestModel = model
                        bestPCA = pca
                        bestScaler = scaler

                # linear Support Vector Machine
                r2, accuracy, model, pca, scaler = classifiers.svcLinear(x,y)
                if(accuracy > bestResult[4]):
                    bestResult[2] = 'SVC (linear)'
                    bestResult[3] = r2
                    bestResult[4] = accuracy
                    bestModel = model
                    bestPCA = pca
                    bestScaler = scaler

                # polynomial Support Vector Machine
                for d in range(1,10):
                    r2, accuracy, model, pca, scaler = classifiers.svcPoly(x,y,d)
                    if(accuracy > bestResult[4]):
                        bestResult[2] = 'SVC (polynomial), Degree: %d' % (d)
                        bestResult[3] = r2
                        bestResult[4] = accuracy
                        bestModel = model
                        bestPCA = pca
                        bestScaler = scaler
                
                # Gaussian Naive Bayes Classifier
                r2, accuracy, model, pca, scaler = classifiers.gaussianNBClassifier(x,y)
                if(accuracy > bestResult[4]):
                    bestResult[2] = 'Gaussian Naive Bayes'
                    bestResult[3] = r2
                    bestResult[4] = accuracy
                    bestModel = model
                    bestPCA = pca
                    bestScaler = scaler

                # Ridge Regression
                r2, accuracy, model, pca, scaler = classifiers.ridgeClassifier(x,y)
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
                
            # append best result to dfBestResult data frame and save model, pca & standardScaler to .pkl files
            dfBestResults.loc[len(dfBestResults)] = globalBestResult
            saveModel(globalBestModel, globalBestPCA, globalBestScaler, t, targetDataFrameName)
            

        else:
            globalBestResult = [t, 'no input feature with p-value below 0.05' ,'-', '-', '-', '-', '-', '-']
            dfBestResults.loc[len(dfBestResults)] = globalBestResult
        
        if (printProgress == True):
            print("completed: " + t)
    
    saveBestResults(dfBestResults, targetDataFrameName)
    return dfBestResults

################################################################################################################################################################
# fill values based on prediction
#def fillValues(x, targetDataFrameName):
#    # initialize values
#    arrayX = x.copy()
#    targetDataFrame = targetDataFrameName.copy()

#    # create input feature
#    if len(arrayX) > 1:
#        inputFeature = arrayX[0]
#        i = 1
#        while i < len(arrayX):
#            inputFeature = pd.merge(inputFeature, arrayX[i], on='user_id', how='inner')
#            i += 1
#    else:
#        inputFeature = arrayX[0]

#    # load bestResults csv
#    dfX = df.copy()
#    dfResult = results.copy()
#    for index, row in dfResult.iterrows():
#        inputFeatureList = row["InputFeature"].split("| ")
#        inputFeature = dfX[inputFeatureList]
#        model = row["Model"]
#        pca = row["PCA"]
#        st_scaler = row["Standard Scaler"]

#        # predict new values
#        x_scaled = st_scaler.transform(inputFeature)
#        x_scaled = pca.transform(x_scaled)
#        dfX[row["TargetFeature"]] = model.predict(x_scaled)
#    return dfX