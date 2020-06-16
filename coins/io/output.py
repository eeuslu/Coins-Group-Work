import pandas as pd
import os
from .utils import get_data_path

def saveInitialDFs(df, fileName):
    path = os.path.join(get_data_path(),'output/initialDataFrames/' + fileName + '.csv')
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
    path = os.path.join(get_data_path(),'output/correlations/significantCorrelations_' + fileName + '.csv')
    df.to_csv(path, sep=';', decimal=',', encoding='utf-8')

def savePreparedDFs(df, fileName):
    path = os.path.join(get_data_path(),'output/preparedDataFrames/' + fileName + '.csv')
    df.to_csv(path, sep=';', decimal=',', encoding='utf-8')