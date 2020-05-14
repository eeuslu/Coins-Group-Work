import json
import numpy as np
import pandas as pd
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, EmotionOptions

# analyses the individual sentiment of each value of a column of a dataframe, both specified by the user and returns a dataframe with an added sentiment column
def analyzeGermanSentiment(df, columnToAnalyze):
    
    # authenticate to and initialize IBM Watson NLU Service
    authenticator_nlu = IAMAuthenticator('4tnw-cTxlGqkOzCyx-5bIMHZD6zH-ENzcoNb3p6IEKJL')
    natural_language_understanding = NaturalLanguageUnderstandingV1(version='2019-07-12', authenticator=authenticator_nlu)
    natural_language_understanding.set_service_url('https://api.eu-de.natural-language-understanding.watson.cloud.ibm.com/instances/63d36bb4-0f8e-45da-aa96-9c1c4ab3b457')

    # empty list for sentiment values
    sentiment = []

    # iterate through every row of the dataframe and ask for sentiment score if cell is not NaN, save sentiment in list
    for index, row in df.iterrows():
        if pd.isnull(row[columnToAnalyze]):
            sentiment.append(np.NaN)
        elif row[columnToAnalyze] == ' ':
            sentiment.append(np.NaN)
        else:
            textToAnalyze = str(row[columnToAnalyze])
            response = natural_language_understanding.analyze(language='de', text=textToAnalyze, features=Features(sentiment=SentimentOptions(document=True))).get_result()
            sentiment_score = response['sentiment']['document']['score']
            sentiment.append(sentiment_score)

    # add all sentiment scores as column to dataframe
    df[columnToAnalyze + '_sentiment'] = sentiment

    # replace empty cells with NaN
    df = df.replace(' ', np.NaN)

    # return dataframe
    return df