from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import coins


def findCorrelations(dfMain,dfList):

    correlationList = []
    rValueList = []
    pValueList = []

    for df in dfList:
        #Get advanced statistics
        #dfRData,dfPData = advancedStatistics(dfMain,df)
        dfRData=1
        dfPData=1

        #Correlation between dfSocialDemographics and dfMotives
        dfMerge = dfMain.set_index('user_id').join(df.set_index('user_id'),how='inner',lsuffix='_l', rsuffix='_r')
        dfMerge = pd.get_dummies(dfMerge)
        dfMergeCorr = dfMerge.corr()

        listMainAttr = []
        for attr in dfMerge.columns:
            if attr in dfMain.columns:
                listMainAttr.append(attr)

    	#Clean Axis
        dfMergeCorrFiltered = dfMergeCorr.drop(listMainAttr,axis=1)
        dfRDataFiltered = dfMergeCorr.drop(dfRData,axis=1)
        dfPDataFiltered = dfMergeCorr.drop(dfPData,axis=1)

        #Clean Rows
        dfMergeCorrFiltered.drop(dfMergeCorrFiltered.columns, inplace=True)
        dfRDataFiltered.drop(dfRDataFiltered.columns,axis=1)
        dfPDataFiltered.drop(dfPDataFiltered.columns,axis=1)

        correlationList.append(dfMergeCorrFiltered)
        rValueList.append(dfRDataFiltered)
        pValueList.append(dfPDataFiltered)

    return correlationList,rValueList,pValueList


def advancedStatistics(dfMain,df):
    
    #Merge and prepare
    dfMerge = dfMain.set_index('user_id').join(df.set_index('user_id'),how='inner',lsuffix='_l', rsuffix='_r')
    dfMerge = pd.get_dummies(dfMerge)
    dfMerge.dropna(inplace=True)

    allRData = []
    allPData = []

    #Go through all combiantions 
    for columnY in dfMerge.columns:
        rowR = []
        rowP = []
        for columnX in dfMerge.columns:
            slope, intercept, r_value, p_value, std_err = linregress(dfMerge[columnY], dfMerge[columnX])
            rowR.append(r_value)
            rowP.append(p_value)
    
        allRData.append(rowR)
        allPData.append(rowP)

    #Add columns & index for r
    dfRData = pd.DataFrame(allRData,columns=dfMerge.columns)
    dfRData.index = dfMerge.columns

    #Add columns & index for p
    dfPData = pd.DataFrame(allPData,columns=dfMerge.columns)
    dfPData.index = dfMerge.columns

    return dfRData,dfPData


#Return all correlation values above
def findHighCorrelation(dfCorrelation, values):

    highCorrelative = []
    for column in dfCorrelation.columns:
        buffer = dfCorrelation[dfCorrelation[column] > abs(values)][column]
        if len(buffer) > 0:
            highCorrelative.append(buffer)

    return highCorrelative


def significantCorrelation(dfPValues,dfRValues,p):

    significantCorrelation = []
    for column in dfPValues.columns:
        row=[]
        for i,value in enumerate(dfPValues[column]):
            if value <= p:
                significantCorrelation.append([column,dfPValues.index[i],dfRValues[column][i+1],value,631])


#Correlation between Social Demographics and Features
def heatmap(dfMergeCorrFiltered):

 plt.subplots(figsize=(25,15))
 sns_plot = sns.heatmap(dfMergeCorrFiltered,  cmap='coolwarm')

 sns_plot.figure.savefig("output.png")
 

def visualizeCorrelation(dfCorrelation, i):

    f = plt.figure(figsize=(25, 50))
    plt.matshow(dfCorrelation,cmap='RdYlGn')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)

    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('first{number}.png'.format(number=i),bbox_inches='tight')   




def reportCorrelation(dfMain, dfNameList, dfList, value):

    correlationList,rValueList,pValueList = findCorrelations(dfMain, dfList)

    for i,result in enumerate(correlationList):

        print("############## {df_name} ###############".format(df_name=dfNameList[i]))
        visualizeCorrelation(result,i)
        
        highCorrelation = findHighCorrelation(result,value)

        for correlation in highCorrelation:
            print("#############".format(df_name=dfNameList[i]))
            print(correlation)
    
    