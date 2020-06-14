import json
import numpy as np
import pandas as pd
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, EmotionOptions
from coins.io import getAPIcredentials

# analyses the individual sentiment and emotion of each value of a column of a dataframe, both specified by the user and returns a dataframe with added sentiment and emotions columns
def analyzeEnglishSentimentAndEmotions(df):
    
    # load authentification
    apiKey = getAPIcredentials('apiKey_ibmWatson')
    apiURL = getAPIcredentials('apiURL_ibmWatson')

    # authenticate to and initialize IBM Watson NLU Service
    authenticator_nlu = IAMAuthenticator(apiKey)
    natural_language_understanding = NaturalLanguageUnderstandingV1(version='2019-07-12', authenticator=authenticator_nlu)
    natural_language_understanding.set_service_url(apiURL)

    # define columns to analyze
    columnsToAnalyze = ['reasons_translation', 'emotions_translation', 'strengths_translation', 'utilization_translation', 'story_translation']

    # iterate over all columns to analyze
    for columnToAnalyze in columnsToAnalyze:
        
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

# fill NaNs in dfImageDescriptions with bfill and ffill where possible, delete other NaN rows
def fillImageDescriptions(df):
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    dfSentiments = df[['reasons_translation_sentiment', 'emotions_translation_sentiment', 'strengths_translation_sentiment', 'utilization_translation_sentiment', 'story_translation_sentiment']]
    dfSadness = df[['reasons_translation_sadness', 'emotions_translation_sadness', 'strengths_translation_sadness', 'utilization_translation_sadness', 'story_translation_sadness']]
    dfJoy = df[['reasons_translation_joy', 'emotions_translation_joy', 'strengths_translation_joy', 'utilization_translation_joy', 'story_translation_joy']]
    dfFear = df[['reasons_translation_fear', 'emotions_translation_fear', 'strengths_translation_fear', 'utilization_translation_fear', 'story_translation_fear']]
    dfDisgust = df[['reasons_translation_disgust', 'emotions_translation_disgust', 'strengths_translation_disgust', 'utilization_translation_disgust', 'story_translation_disgust']]
    dfAnger = df[['reasons_translation_anger', 'emotions_translation_anger', 'strengths_translation_anger', 'utilization_translation_anger', 'story_translation_anger']]

    dfSentiments = dfSentiments.fillna(method='ffill', axis='columns')
    dfSentiments = dfSentiments.fillna(method='bfill', axis='columns')
    dfSadness = dfSadness.fillna(method='ffill', axis='columns')
    dfSadness = dfSadness.fillna(method='bfill', axis='columns')
    dfJoy = dfJoy.fillna(method='ffill', axis='columns')
    dfJoy = dfJoy.fillna(method='bfill', axis='columns')
    dfFear = dfFear.fillna(method='ffill', axis='columns')
    dfFear = dfFear.fillna(method='bfill', axis='columns')
    dfDisgust = dfDisgust.fillna(method='ffill', axis='columns')
    dfDisgust = dfDisgust.fillna(method='bfill', axis='columns')
    dfAnger = dfAnger.fillna(method='ffill', axis='columns')
    dfAnger = dfAnger.fillna(method='bfill', axis='columns')
    
    df[['reasons_translation_sentiment', 'emotions_translation_sentiment', 'strengths_translation_sentiment', 'utilization_translation_sentiment', 'story_translation_sentiment']] = dfSentiments
    df[['reasons_translation_sadness', 'emotions_translation_sadness', 'strengths_translation_sadness', 'utilization_translation_sadness', 'story_translation_sadness']] = dfSadness
    df[['reasons_translation_joy', 'emotions_translation_joy', 'strengths_translation_joy', 'utilization_translation_joy', 'story_translation_joy']] = dfJoy
    df[['reasons_translation_fear', 'emotions_translation_fear', 'strengths_translation_fear', 'utilization_translation_fear', 'story_translation_fear']] = dfFear
    df[['reasons_translation_disgust', 'emotions_translation_disgust', 'strengths_translation_disgust', 'utilization_translation_disgust', 'story_translation_disgust']] = dfDisgust
    df[['reasons_translation_anger', 'emotions_translation_anger', 'strengths_translation_anger', 'utilization_translation_anger', 'story_translation_anger']] = dfAnger
    
    df = df.dropna(axis='index', subset=['reasons_translation_sentiment', 'reasons_translation_sadness', 'reasons_translation_joy', 'reasons_translation_fear', 'reasons_translation_disgust', 'reasons_translation_anger', 'emotions_translation_sentiment', 'emotions_translation_sadness', 'emotions_translation_joy', 'emotions_translation_fear', 'emotions_translation_disgust', 'emotions_translation_anger', 'strengths_translation_sentiment', 'strengths_translation_sadness', 'strengths_translation_joy', 'strengths_translation_fear', 'strengths_translation_disgust', 'strengths_translation_anger', 'utilization_translation_sentiment', 'utilization_translation_sadness', 'utilization_translation_joy', 'utilization_translation_fear', 'utilization_translation_disgust', 'utilization_translation_anger', 'story_translation_sentiment', 'story_translation_sadness', 'story_translation_joy', 'story_translation_fear', 'story_translation_disgust', 'story_translation_anger'])
    
    df = df.reset_index(drop=True)
    
    return df