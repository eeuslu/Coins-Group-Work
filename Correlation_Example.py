#%%
!pip install -e .

import coins
import pandas as pd

#%%
#First get the raw data
df1 = coins.io.get_data('ipip')
df2 = coins.io.get_data('mpzm')
df3 = coins.io.get_data('images')
df4 = coins.io.get_data('emotions')
df5 = coins.io.get_data('mood')

#Then create the dataframes 
dfPersonality = coins.dfcreation.createPersonality(df1)
dfSocialDemographics = coins.dfcreation.createSocialDemographics(df1,df2,df3,df4,df5)
dfImageRating = coins.dfcreation.createImageRatings(df3)


#CURRENTLY NOT IN USE
dfMotives = coins.dfcreation.createMotives(df2)
dfMood = coins.dfcreation.createMood(df5)


#%%
#Get the sentiment Score
dfImageDescriptions = pd.read_csv('all_sentimentScores.csv')
dfImageDescriptions = coins.dfcreation.cleanImageDescriptions(dfImageDescriptions)
dfImageDescriptions.drop(['reasons', 'emotions', 'strengths', 'utilization', 'story'], axis=1, inplace=True)
#%%
#Small example how to filter for the scores for a destinct image
dfImageDescriptionsFiltered = dfImageDescriptions[dfImageDescriptions['file_name']=='erwachsene-pixabay-01_gipfel1.jpg']
dfImageDescriptionsFiltered.drop('file_name',axis=1,inplace=True)
dfImageDescriptionsFiltered

#%%
#Get the report (4 parameters needed)
#1. First overgive the "main" dataframe -- all the following dataframes refer to him    --type=DataFrame
#2. Overgive the names of the other dataframes     --type=list(string)
#3. Overgive the other dataframes                  --type=list(DataFrame)
#4. Overgive a value. All correlations higher then this value are displayed.

coins.dataExploration.reportCorrelation(dfImageDescriptionsFiltered,['dfSocialDemographics','dfPersonality','dfImageRating'],[dfSocialDemographics,dfPersonality,dfImageRating],0.2)
print('#################################1#######################################')


#OUTPUT:
#The correlation of:
# dfImageDescriptionsFiltered <-> dfSocialDemographics
# dfImageDescriptionsFiltered <-> dfPersonality
# dfImageDescriptionsFiltered <-> dfImageRating

# For this you get a visualisation first
# Then (if existing) the correlations higher then your overgiven value








#%%
coins.dataExploration.reportCorrelation(dfPersonality,['dfSocialDemographics','dfImageRating','dfMotives','dfMood'],[dfSocialDemographics,dfImageRating,dfMotives,dfMood],0.2)
print('#################################2#######################################')
#%%
coins.dataExploration.reportCorrelation(dfImageRating,['dfSocialDemographics','dfPersonality','dfMotives','dfMood'],[dfSocialDemographics,dfPersonality,dfMotives,dfMood],0.2)
print('#################################3#######################################')
#%%
coins.dataExploration.reportCorrelation(dfMotives,['dfSocialDemographics','dfPersonality','dfImageRating','dfMood'],[dfSocialDemographics,dfPersonality,dfImageRating,dfMood],0.2)
print('#################################4#######################################')
#%%
coins.dataExploration.reportCorrelation(dfMood,['dfSocialDemographics','dfPersonality','dfImageRating','dfMotives'],[dfSocialDemographics,dfPersonality,dfImageRating,dfMotives],0.3)
print('#################################4#######################################')
