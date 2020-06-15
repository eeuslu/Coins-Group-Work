import pandas as pd

#Prepare the SocioDemographic data for the correlation finding
#This include to drop redundanct columns and drop seldom values
def prepare_SocioDemographics(dfSocioDemographics, dropLowerThenProzent):

    #Make a copy first
    dfSocioDemographicsCopy = dfSocioDemographics.copy()
    # drop some columns with high redundance
    dfSocioDemographicsCopy.drop(['registration_ageKat', 'work_country', 'work_district', 'job_status_retired'] ,axis=1, inplace=True)
    dfSocioDemographicsCopy.dropma(inplace=True)

    #Get original columns
    originColumns = dfSocioDemographicsCopy.columns

    #Save the IDS and drop it
    dfSocialDemographicsID = dfSocioDemographicsCopy['user_id']
    dfSocioDemographicsCopy.drop('user_id',axis=1,inplace=True)

    #Create dummies
    dfSocialDemographicsDummies = pd.get_dummies(dfSocioDemographicsCopy)

    #Get all non-Dummie columns -> drop originColumns
    dummieColums = dfSocialDemographicsDummies.drop(originColumns,axis=1,errors='ignore').columns

    #Drop all dummie collumns where dropLowerThenProzent taken place
    originalLength = len(dfSocialDemographicsDummies)
    dropedColumnsList = []

    for column in dummieColums:
        #Get number of columns where dummi is 1
        length = len(dfSocialDemographicsDummies[dfSocialDemographicsDummies[column] == 1])
        
        #Get percentage where dummie is 1. If lower then overgiven value drop
        if (length/originalLength) < (dropLowerThenProzent/100):
            dfSocialDemographicsDummies.drop(column,axis=1,inplace=True)
            dropedColumnsList.append(column)

    #Drop all columns with

    #Create a second version include the user ids
    dfSocialDemographicsDummiesWithID = dfSocialDemographicsDummies.copy()
    dfSocialDemographicsDummiesWithID['user_id'] = dfSocialDemographicsID
  
    #Drop na values
    dfSocialDemographicsDummies.dropna(inplace=True)
    dfSocialDemographicsDummiesWithID.dropna(inplace=True)

    return dfSocialDemographicsDummies,dfSocialDemographicsDummiesWithID,dropedColumnsList


# transform values in dfPersonality from real numbers to classes for classification purpose
def preparePersonality(dfPersonality, multiclass=False, split="hard"):

    dfInput = dfPersonality.copy()
    
    # check if multiclass prediction is wanted
    if multiclass == False:

        # go through all columns and make the split
        for column in dfPersonality.columns:
            if column != "user_id":
                
                # define new column name
                colName = column+"Category"

                # check for split type
                if split == "hard":
                    border = 3
                elif split == "mean":
                    border = dfInput[column].mean()
                elif split == "median":
                    border = dfInput[column].median()

                # split
                dfInput.loc[dfInput[column] < border, colName] = 0
                dfInput.loc[dfInput[column] >= border, colName] = 1

                # drop old column
                dfInput.drop(column, axis=1, inplace=True)
    
    elif multiclass == True:

        # go through all columns and make the split
        for column in dfPersonality.columns:
            if column != "user_id":
                
                # define new column name
                colName = column+"Category"

                # check for split type
                if split == "hard":
                    border1 = 2.5
                    border2 = 3.5
                elif split == 'thirds':
                    border1 = dfInput[column].quantile(0.33)
                    border2 = dfInput[column].quantile(0.66)

                # split
                dfInput.loc[dfInput[column] < border1, colName] = 0
                dfInput.loc[(dfInput[column] >= border1) & (dfInput[column] < border2), colName] = 1
                dfInput.loc[dfInput[column] >= border2, colName] = 2

                # drop old column
                dfInput.drop(column, axis=1, inplace=True)

    return dfInput


def prepareImageDescriptions(dfImageDescriptions, multiclass=False, split='median'):
    
    # drop unnecessary columns and all NaN values
    dfImageDescriptions = dfImageDescriptions.drop(columns=['file_name', 'reasons', 'emotions', 'strengths', 'utilization', 'story', 'reasons_translation', 'emotions_translation', 'strengths_translation', 'utilization_translation', 'story_translation'])
    dfImageDescriptions = dfImageDescriptions.dropna(axis='index')

    dfInput = dfImageDescriptions.copy()

    # check if multiclass prediction is wanted
    if multiclass == False:

        # go through all columns and make the split
        for column in dfImageDescriptions:
            if column != "user_id":
                
                # define new column name
                colName = column+"Category"

                # check for split type
                if split == 'hard':
                    # check if column contains sentiment or emotion value
                    if 'sentiment' in column:
                        border = 0                        
                    elif (('reasons' in column) | ('strengths' in column) | ('emotions' in column) | ('utilization' in column) | ('story' in column)):
                        border = 0.5
                elif split == 'mean':
                    border = dfInput[column].mean()
                elif split == 'median':
                    border = dfInput[column].median()
                
                # split
                dfInput.loc[dfInput[column] < border, colName] = 0
                dfInput.loc[dfInput[column] >= border, colName] = 1

                # drop old column
                dfInput.drop(column, axis=1, inplace=True)
            
    elif multiclass == True:

        # go through all columns and make the split
        for column in dfImageDescriptions:
            if column != "user_id":
                
                # define new column name
                colName = column+"Category"

                # check for split type
                if split == 'hard':
                    # check if column contains sentiment or emotion value
                    if 'sentiment' in column:
                        border1 = -0.25        
                        border2 = 0.25                
                    elif (('reasons' in column) | ('strengths' in column) | ('emotions' in column) | ('utilization' in column) | ('story' in column)):
                        border1 = 0.33
                        border2 = 0.66
                elif split == 'thirds':
                    border1 = dfInput[column].quantile(0.33)
                    border2 = dfInput[column].quantile(0.66)
                
                # split
                dfInput.loc[dfInput[column] < border1, colName] = 0
                dfInput.loc[(dfInput[column] >= border1) & (dfInput[column] < border2), colName] = 1
                dfInput.loc[dfInput[column] >= border2, colName] = 2

                # drop old column
                dfInput.drop(column, axis=1, inplace=True)

    return dfInput
