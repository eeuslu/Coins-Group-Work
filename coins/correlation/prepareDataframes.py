import pandas as pd
from sklearn.cluster import KMeans
from .utils import get_data_path

# general cleaning of dfSocioDemographics, transforming categorical values using One Hot Encoding, drop categories with low sample size
def prepareSocioDemographics(dfSocioDemographics, dropPercentage):
    
    # drop NaN values
    dfSocioDemographics.dropna(inplace=True)

    # get original columns
    originalColumns = dfSocioDemographics.columns

    # save the user IDs and drop them
    dfSocioDemographicsID = dfSocioDemographics['user_id']
    dfSocioDemographics.drop('user_id',axis=1,inplace=True)

    # create dummies
    dfSocioDemographicsDummies = pd.get_dummies(dfSocioDemographics)

    # get all binary columns
    binaryColumns = dfSocioDemographicsDummies.drop(originalColumns,axis=1,errors='ignore').columns.tolist()
    jobStatusColumns = ['job_status_edu_parttime', 'job_status_edu_fulltime', 'job_status_employed_parttime', 'job_status_employed_fulltime', 'job_status_selfemployed', 'job_status_houskeeping', 'job_status_unemployed', 'job_status_retired']
    for column in jobStatusColumns:
        binaryColumns.append(column)

    # drop all dummy columns where sample size is smaller than dropPercentage
    originalLength = len(dfSocioDemographicsDummies)
    droppedColumnsList = []

    for column in binaryColumns:
        
        # get number of rows where dummy is 1
        length = len(dfSocioDemographicsDummies[dfSocioDemographicsDummies[column] == 1])
        
        # get percentage where dummy is 1. Drop if lower than dropPercentage
        if (length/originalLength) < (dropPercentage/100):
            dfSocioDemographicsDummies.drop(column,axis=1,inplace=True)
            droppedColumnsList.append(column)

    # add user IDs again
    #dfSocioDemographicsDummies['user_id'] = dfSocioDemographicsID
    dfSocioDemographicsDummies.insert(loc=0, column='user_id', value=dfSocioDemographicsID)
    dfSocioDemographicsDummies = dfSocioDemographicsDummies.reset_index(drop=True)
  
    # drop NaN values
    dfSocioDemographicsDummies.dropna(inplace=True)

    # save all columns
    columns = dfSocioDemographicsDummies.columns
    dfColumns = pd.DataFrame(columns)
    path = '{directory}/input/SystemOwned/dfSocioDemographics.csv'.format(directory=get_data_path())
    dfColumns.to_csv(path, sep=";", index=False, header=False)

    return dfSocioDemographicsDummies, droppedColumnsList


def cluster(dfInput,cluster=5, onlyCluster=False):
    
    dfWithId = dfInput.copy()

    if cluster > 0:
        dfWithoutId = dfWithId.drop('user_id',axis=1)

        kmeans = KMeans(n_clusters=cluster, random_state=0).fit(dfWithoutId)

        prediction = kmeans.predict(dfWithoutId)
        dfWithId['cluster'] = prediction
        dfWithId['cluster'] = dfWithId['cluster'].astype('float')

        if onlyCluster == True:
            for column in dfWithId.columns:
                if column not in ['user_id','cluster']:
                    dfWithId.drop(column,axis=1,inplace=True)

    return dfWithId



# transform values in dfPersonality from real numbers to classes for classification purpose
def preparePersonality(dfPersonality,train, multiclass=False, split="median"):

    #For saving
    path = '{directory}/input/SystemOwned/bordersPersonality.csv'.format(directory=get_data_path())
    #empty list for configuration saving
    configuration = []
    #Copy
    dfInput = dfPersonality.copy()


    if train == False:
        dfConfiguration = pd.read_csv(path,sep=';',header=None)
        dfConfigurationValues = dfConfiguration.values
        
        multiclass = dfConfigurationValues[0][0]


    #Decide how many splits should occure
    if  multiclass == False:
        count=0
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


                if train == False:
                    border = dfConfigurationValues[count][1]
                    count +=1 

                #Save borders
                configuration.append([multiclass,border,0])

                # split
                dfInput.loc[dfInput[column] < border, colName] = 0
                dfInput.loc[dfInput[column] >= border, colName] = 1

                # drop old column
                dfInput.drop(column, axis=1, inplace=True)

     


    #Decide how many splits should occure
    elif  multiclass == True:
        count=0
        #Go through all columns and make the split
        for column in dfPersonality.columns:
            if column != "user_id":
                
                # define new column name
                colName = column+"Category"

                # check for split type
                if split == "hard":
                    border1 = 2.5
                    border2 = 3.5
                else:
                    border1 = dfInput[column].quantile(0.33)
                    border2 = dfInput[column].quantile(0.66)

                if train == False:
                    border1 = dfConfigurationValues[count][1]
                    border2 = dfConfigurationValues[count][2]
                    count +=1 

                #Save borders
                configuration.append([multiclass,border1,border2])

                # split
                dfInput.loc[dfInput[column] < border1, colName] = 0
                dfInput.loc[(dfInput[column] >= border1) & (dfInput[column] < border2), colName] = 1
                dfInput.loc[dfInput[column] >= border2, colName] = 2

                # drop old column
                dfInput.drop(column, axis=1, inplace=True)


    if train == True:
        dfConfiguration = pd.DataFrame(configuration)
        dfConfiguration.to_csv(path,sep=';',header=False, index=False)

    return dfInput




def prepareImageDescriptions(dfImageDescriptions,train, multiclass=False, split='median'):
    #For saving
    path = '{directory}/input/SystemOwned/bordersImageDescription.csv'.format(directory=get_data_path())
    #empty list for configuration saving
    configuration = []
    
    # drop unnecessary columns and all NaN values
    dfImageDescriptions = dfImageDescriptions.drop(columns=['file_name', 'reasons', 'emotions', 'strengths', 'utilization', 'story', 'reasons_translation', 'emotions_translation', 'strengths_translation', 'utilization_translation', 'story_translation'])
    dfImageDescriptions = dfImageDescriptions.dropna(axis='index')

    dfInput = dfImageDescriptions.copy()

    if train == False:
        dfConfiguration = pd.read_csv(path,sep=';',header=None)
        dfConfigurationValues = dfConfiguration.values
        
        multiclass = dfConfigurationValues[0][0]

    # check if multiclass prediction is wanted
    if multiclass == False:

        # go through all columns and make the split
        count = 0
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

                
                if train == False:
                    border = dfConfigurationValues[count][1]
                    count +=1 

                #Save borders
                configuration.append([multiclass,border,0])

                # split
                dfInput.loc[dfInput[column] < border, colName] = 0
                dfInput.loc[dfInput[column] >= border, colName] = 1

                # drop old column
                dfInput.drop(column, axis=1, inplace=True)
            
    elif multiclass == True:
        count = 0
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
                else:
                    border1 = dfInput[column].quantile(0.33)
                    border2 = dfInput[column].quantile(0.66)
                

                if train == False:
                    border1 = dfConfigurationValues[count][1]
                    border2 = dfConfigurationValues[count][2]
                    count +=1 

                #Save borders
                configuration.append([multiclass,border1,border2])

                # split
                dfInput.loc[dfInput[column] < border1, colName] = 0
                dfInput.loc[(dfInput[column] >= border1) & (dfInput[column] < border2), colName] = 1
                dfInput.loc[dfInput[column] >= border2, colName] = 2

                # drop old column
                dfInput.drop(column, axis=1, inplace=True)

    if train == True:
        dfConfiguration = pd.DataFrame(configuration)
        dfConfiguration.to_csv(path,sep=';',header=False, index=False)
        

    return dfInput
