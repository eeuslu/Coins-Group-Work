import pandas as pd

def balanceAccordingToColumn(dfInput,balnceColumn):

    #Create an empty dataframe to fill it up
    dfOutput = pd.DataFrame(None,columns=dfInput.columns)

    #Check if column exists
    if balnceColumn in dfInput.columns:

        #Get lowest number 
        minValue = min(dfInput[balnceColumn].value_counts().to_list())
        values = dfInput[balnceColumn].unique()

        for value in values:
            dfForValue = dfInput[dfInput[balnceColumn] == value].sample(n=minValue)
            dfOutput = pd.concat([dfOutput,dfForValue])

        for column in dfOutput.columns:
            if column != 'user_id':
                if dfOutput[column].dtype == 'object':
                    dfOutput[column] = pd.to_numeric(dfOutput[column])

        return dfOutput
    
    else:
        print("Column does not exist in Dataframe")

