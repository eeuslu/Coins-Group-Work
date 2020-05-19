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


# analyses the individual sentiment and emotion of each value of a column of a dataframe, both specified by the user and returns a dataframe with added sentiment and emotions columns
def analyzeEnglishSentimentAndEmotions(df, columnToAnalyze):
    
    # authenticate to and initialize IBM Watson NLU Service
    authenticator_nlu = IAMAuthenticator('4tnw-cTxlGqkOzCyx-5bIMHZD6zH-ENzcoNb3p6IEKJL')
    natural_language_understanding = NaturalLanguageUnderstandingV1(version='2019-07-12', authenticator=authenticator_nlu)
    natural_language_understanding.set_service_url('https://api.eu-de.natural-language-understanding.watson.cloud.ibm.com/instances/63d36bb4-0f8e-45da-aa96-9c1c4ab3b457')

    # empty lists for sentiment and emotion values
    sentiment = []
    sadness = []
    joy = []
    fear = []
    disgust = []
    anger = []

    # iterate through every row of the dataframe and ask for sentiment and emotion scores if cell is not NaN, save sentiments and emotions in lists
    for index, row in df.iterrows():
        if pd.isnull(row[columnToAnalyze]):
            sentiment.append(np.NaN)
            sadness.append(np.NaN)
            joy.append(np.NaN)
            fear.append(np.NaN)
            disgust.append(np.NaN)
            anger.append(np.NaN)
        elif row[columnToAnalyze] == ' ':
            sentiment.append(np.NaN)
            sadness.append(np.NaN)
            joy.append(np.NaN)
            fear.append(np.NaN)
            disgust.append(np.NaN)
            anger.append(np.NaN)
        else:
            textToAnalyze = str(row[columnToAnalyze])
            response = natural_language_understanding.analyze(language='en', text=textToAnalyze, features=Features(sentiment=SentimentOptions(document=True), emotion=EmotionOptions(document=True))).get_result()
            sentiment_score = response['sentiment']['document']['score']
            sadness_score = response['emotion']['document']['emotion']['sadness']
            joy_score = response['emotion']['document']['emotion']['joy']
            fear_score = response['emotion']['document']['emotion']['fear']
            disgust_score = response['emotion']['document']['emotion']['disgust']
            anger_score = response['emotion']['document']['emotion']['anger']
            sentiment.append(sentiment_score)
            sadness.append(sadness_score)
            joy.append(joy_score)
            fear.append(fear_score)
            disgust.append(disgust_score)
            anger.append(anger_score)

    # add all sentiment scores as column to dataframe
    df[columnToAnalyze + '_sentiment'] = sentiment
    df[columnToAnalyze + '_sadness'] = sadness
    df[columnToAnalyze + '_joy'] = joy
    df[columnToAnalyze + '_fear'] = fear
    df[columnToAnalyze + '_disgust'] = disgust
    df[columnToAnalyze + '_anger'] = anger

    # replace empty cells with NaN
    df = df.replace(' ', np.NaN)

    # return dataframe
    return df