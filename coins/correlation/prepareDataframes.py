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


#To make later steps easier, use value ranges instead of real numbers.
#As an alternative use clusters.
def prepare_Personality(dfPersonality, multiclass=False, split="Median", cluster=0):

    dfInput = dfPersonality.copy()
    #Decide how many splits should occure
    if  multiclass == False:

        #Go through all columns and make the split
        for column in dfPersonality.columns:
            if column != "user_id":
                colName = column+"Category"

                if split == "Hard":
                    dfInput.loc[dfInput[column] < 3, colName] = 0
                    dfInput.loc[dfInput[column] >= 3, colName] = 1
                    
                elif split == "Mean":
                    border = dfInput[column].mean()
                    dfInput.loc[dfInput[column] < border, colName] = 0
                    dfInput.loc[dfInput[column] >= border, colName] = 1
                
                elif split == "Median":
                    border = dfInput[column].median()
                    dfInput.loc[dfInput[column] < border, colName] = 0
                    dfInput.loc[dfInput[column] >= border, colName] = 1


                dfInput.drop(column, axis=1, inplace=True)


    #Decide how many splits should occure
    if  multiclass == True:

        #Go through all columns and make the split
        for column in dfPersonality.columns:
            if column != "user_id":
                colName = column+"Category"

                if split == "Hard":
                    dfInput.loc[dfInput[column] < 2.5, colName] = 0
                    dfInput.loc[(dfInput[column] >= 2.5) & (dfInput[column] < 3.5), colName] = 1
                    dfInput.loc[dfInput[column] >= 3.5, colName] = 2

                else:
                    quantilU = dfInput[column].quantile(0.33)
                    quantilO = dfInput[column].quantile(0.66)

                    dfInput.loc[dfInput[column] < quantilU, colName] = 0
                    dfInput.loc[(dfInput[column] >= quantilU) & (dfInput[column] < quantilO), colName] = 1
                    dfInput.loc[dfInput[column] >= quantilO, colName] = 2
                dfInput.drop(column, axis=1, inplace=True)

    return dfInput




def imageDescriptionToCategory(df):
    dfInput = df.copy()
    for column in dfInput:
        if column != "user_id":
            colName = column+"Category"
            if "sentiment" in column:
                #firstBorder = 0
                #secondBorder = dfInput[column].quantile(0.66)
                dfInput.loc[dfInput[column] < -0.25, colName] = -1
                #dfInput.loc[dfInput[column].round() == 0, colName] = 0
                #dfInput.loc[dfInput[column].round() == 1, colName] = 1
                dfInput.loc[((-0.25 <= dfInput[column]) & (dfInput[column] < 0.25)), colName] = 0
                dfInput.loc[dfInput[column] >= 0.25, colName] = 1
                dfInput.drop(column, axis=1, inplace=True)
            elif (('reasons' in column) | ('strengths' in column) | ('emotions' in column) | ('utilization' in column) | ('story' in column)): 
                firstBorder = dfInput[column].quantile(0.33)
                secondBorder = dfInput[column].quantile(0.66)
                dfInput.loc[dfInput[column] < firstBorder, colName] = -1
                dfInput.loc[((firstBorder <= dfInput[column]) & (dfInput[column] < secondBorder)), colName] = 0
                dfInput.loc[dfInput[column] >= secondBorder, colName] = 1
                dfInput.drop(column, axis=1, inplace=True)
    return dfInput
