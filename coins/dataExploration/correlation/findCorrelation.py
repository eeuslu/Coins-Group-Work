import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import coins


def findCorrelations(dfSocialDemographics,dfPersonality,dfImageRatings,dfMotives,dfMood):

    dfMerge1 = dfSocialDemographics.set_index('user_id').join(dfPersonality.set_index('user_id'),how='inner',lsuffix='_l', rsuffix='_r')
    dfMerge1 = pd.get_dummies(dfMerge1)
    dfMerge1Corr = dfMerge1.corr()

    attributSet1 = ['neurotizismus','extraversion','offenheit','vertraeglichkeit','gewissenhaftigkeit']
    dfMerge1CorrFiltered = dfMerge1Corr.drop(attributSet1,axis=1)

    attributSet2 = list(dfMerge1CorrFiltered.columns)
    dfMerge1CorrFiltered = dfMerge1CorrFiltered.drop(attributSet2)

    return dfMerge1CorrFiltered

#Return all values above
def findHighCorrelation(dfCorrelation, values):

    highCorrelative = []
    for column in dfCorrelation.columns:
        buffer = dfCorrelation[dfCorrelation[column] > abs(values)][column]
        if len(buffer) > 0:
            highCorrelative.append(buffer)

    return highCorrelative

# def heatmap(dfCorrelation):

#     corr = dfMerge1CorrFiltered.corr()
#     x = sns.heatmap(
#     corr, 
#     vmin=-1, vmax=1, center=0,
#     cmap=sns.diverging_palette(20, 220, n=200),
#     square=True
# )
#     ax.set_xticklabels(
#     ax.get_xticklabels(),
#     rotation=45,
#     horizontalalignment='right'
# );

#     return heatmap


def visualizeCorrelation(dfCorrelation):

    f = plt.figure(figsize=(25, 50))
    plt.matshow(dfCorrelation)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)

    #fig1 = plt.gcf()
    plt.show()
    plt.draw()
    #f.savefig('first.png',bbox_inches='tight')   
