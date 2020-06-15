import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr

def calculateCorrWithPValue(x,y):
    dfX = x.copy()
    dfY = y.copy()
    # join both datasets on user_id
    dfJoinedData = pd.merge(dfY, dfX, on='user_id', how='inner')
    # only consider users which have data in personality and socialDemographics
    xLength = len(dfX.columns)
    yLength = len(dfY.columns)
    dfY = dfJoinedData.iloc[:,1:yLength]
    dfX = dfJoinedData.iloc[:,yLength:]

    # calculate pValue
    pValue = calculatePValue(dfY, dfX)
    pValue = pValue.apply(pd.to_numeric, errors='coerce')

    # calculate correlation
    correlation = dfJoinedData.corr()
    #correlation = correlation.iloc[0:(yLength-1),(yLength-1):]

    return dfX, dfY, dfJoinedData, pValue, correlation

# calculate p-values for the connection of each column of two data frames
def calculatePValue(x, y):
    x = x.dropna()._get_numeric_data()
    y = y.dropna()._get_numeric_data()
    xCols = pd.DataFrame(columns=x.columns)
    yCols = pd.DataFrame(columns=y.columns)
    pvalues = xCols.transpose().join(yCols, how='outer')
    for r in y.columns:
        for c in x.columns:
            pvalues[r][c] = round(pearsonr(y[r], x[c])[1], 4)
    return pvalues