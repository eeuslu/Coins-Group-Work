import pandas as pd

def balanceAccordingToColumn(dfInput,balnceColumn):

    #Create an empty dataframe to fill it up
    dfPOutput = pd.DataFrame(None,columns=dfInput.columns)

    #Check if column exists
    if balnceColumn in dfInput.columns:

        #Get lowest number 
        minValue = min(dfInput[balnceColumn].value_counts().to_list())
        values = dfInput[balnceColumn].unique()

        for value in values:
            dfForValue = dfInput[dfInput[balnceColumn] == value].sample(n=minValue)
            dfPOutput = pd.concat([dfPOutput,dfForValue])

        return dfPOutput
    
    else:
        print("Column does not exist in Dataframe")

