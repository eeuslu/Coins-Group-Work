import pandas as pd
import os
from .utils import get_data_path
from sklearn.externals.joblib import dump, load
import hashlib

def saveInitialDFs(df, fileName):

    path = '{directory}/output/initialDataFrames/{filename}.csv'.format(directory=get_data_path(),filename=fileName)
    df.to_csv(path, sep=';', decimal=',', encoding='utf-8')



def saveAnalyzedImageDescriptions(df):

    path = os.path.join(get_data_path(),'output/analyzedDataFrames/analyzedImageDescriptions.csv')
    df.to_csv(path, sep=';', decimal=',', encoding='utf-8')



# def saveCorrelationsAndPValues(dfC, dfP, fileName):
#     pathC = os.path.join(get_data_path(),'output/correlations/c_' + fileName + '.csv')
#     dfC.to_csv(pathC, sep=';', decimal=',', encoding='utf-8')
#     pathP = os.path.join(get_data_path(),'output/correlations/p_' + fileName + '.csv')
#     dfP.to_csv(pathP, sep=';', decimal=',', encoding='utf-8')



def saveSignificantCorrelations(df, fileName):

    path = '{directory}/output/correlations/significantCorrelations_{filename}.csv'.format(directory=get_data_path(),filename=fileName)
    df.to_csv(path, sep=';', decimal=',', encoding='utf-8')



def savePreparedDFs(df, fileName):

    path = '{directory}/output/preparedDataFrames/{filename}.csv'.format(directory=get_data_path(),filename=fileName)
    df.to_csv(path, sep=';', decimal=',', encoding='utf-8')



def saveModel(model, pca, standardScaler, targetFeatureName, targetDataFrameName):

    targetFeatureName = getHash(targetFeatureName)

    #Save the model
    path = '{directory}/output/modelResults/{target}/model/{feature}.pkl'.format(directory=get_data_path(),target=targetDataFrameName, feature=targetFeatureName)
    dump(model,path)

    #Save the PCA
    path = '{directory}/output/modelResults/{target}/pca/{feature}.pkl'.format(directory=get_data_path(),target=targetDataFrameName, feature=targetFeatureName)
    dump(pca,path)

    #Save the StandardScaler
    path = '{directory}/output/modelResults/{target}/scaler/{feature}.pkl'.format(directory=get_data_path(),target=targetDataFrameName, feature=targetFeatureName)
    dump(standardScaler,path)



def saveBestResults(df, targetDataFrameName):
    path = '{directory}/output/modelResults/bestResults/{filename}.csv'.format(directory=get_data_path(),filename=targetDataFrameName)
    df.to_csv(path, sep=';', decimal=',', encoding='utf-8')



def getHash(name):
    return hashlib.sha256(name.encode()).hexdigest()[:20]