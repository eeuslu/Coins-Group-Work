import matplotlib.pyplot as plt
import pandas as pd
import coins


def findCorrelations(dfMain,dfList):

    correlationList = []

    for df in dfList:

        #Correlation between dfSocialDemographics and dfMotives
        dfMerge = dfMain.set_index('user_id').join(df.set_index('user_id'),how='inner',lsuffix='_l', rsuffix='_r')
        dfMerge = pd.get_dummies(dfMerge)
        dfMergeCorr = dfMerge.corr()

        attributSet1 = list(df.drop('user_id',axis=1).columns)
        dfMergeCorrFiltered = dfMergeCorr.drop(attributSet1,axis=1)

        attributSet2 = list(dfMergeCorrFiltered.columns)
    
        #Final df
        dfMergeCorrFiltered = dfMergeCorrFiltered.drop(attributSet2)
        correlationList.append(dfMergeCorrFiltered)

    return correlationList


#Return all values above
def findHighCorrelation(dfCorrelation, values):

    highCorrelative = []
    for column in dfCorrelation.columns:
        buffer = dfCorrelation[dfCorrelation[column] > abs(values)][column]
        if len(buffer) > 0:
            highCorrelative.append(buffer)

    return highCorrelative



def visualizeCorrelation(dfCorrelation, i):

    f = plt.figure(figsize=(25, 50))
    plt.matshow(dfCorrelation)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)

    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('first{number}.png'.format(number=i),bbox_inches='tight')   



def reportCorrelation(dfMain, dfNameList, dfList, value):

    correlationList = findCorrelations(dfMain, dfList)

    for i,result in enumerate(correlationList):

        print("############## {df_name} ###############".format(df_name=dfNameList[i]))
        visualizeCorrelation(result,i)
        highCorrelation = findHighCorrelation(result,value)

        for correlation in highCorrelation:
            print("#############".format(df_name=dfNameList[i]))
            print(correlation)
