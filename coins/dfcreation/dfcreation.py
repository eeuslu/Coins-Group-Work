import pandas as pd
import numpy as numpy

def createPersonality(df_ipip):

    # get important columns from ipip dataframe
    dfPersonality = df_ipip[['user_id','N','E','O','A','C']]
    dfPersonality.columns=['user_id','neurotizismus','extraversion','offenheit','vertraeglichkeit','gewissenhaftigkeit']

    #drop duplicates
    dfPersonality.drop_duplicates(inplace=True)

    dfPersonality = dfPersonality.reset_index(drop=True)

    return dfPersonality


def createSocioDemographics(df_ipip, df_mpzm, df_images, df_mood):
    
    sources = [df_mpzm, df_images, df_mood]
    
    # create first DataFrame as base for concat
    dfSocialDemographics = df_ipip[['user_id','gender','registration_age','registration_ageKat','country','work_country','work_district',
                            'job_position','job_sector','company_size','job_status_edu_parttime','job_status_edu_fulltime',
                            'job_status_employed_parttime','job_status_employed_fulltime','job_status_selfemployed',
                            'job_status_houskeeping','job_status_unemployed','job_status_retired','educational_achievement']]

    # concat defined columns with all other dataframes containing valuable information about demographics
    for source in sources:
        dfSocialDemographicsSource = source[['user_id','gender','registration_age','registration_ageKat','country','work_country','work_district',
                            'job_position','job_sector','company_size','job_status_edu_parttime','job_status_edu_fulltime',
                            'job_status_employed_parttime','job_status_employed_fulltime','job_status_selfemployed',
                            'job_status_houskeeping','job_status_unemployed','job_status_retired','educational_achievement']]
        dfSocialDemographics = pd.concat([dfSocialDemographics,dfSocialDemographicsSource],ignore_index=True)
    
    # drop duplicates
    dfSocialDemographics.drop_duplicates(inplace=True)

    # reset index
    dfSocialDemographics = dfSocialDemographics.reset_index(drop=True)

    return dfSocialDemographics
    

def createImageDescriptions(df_images):

    # get important columns from images dataframe
    dfTextAboutImages = df_images[['user_id','file_name','reasons','emotions','strengths','utilization','story','favorite']]
    
    # drop duplicates
    dfTextAboutImages.drop_duplicates(inplace=True)
    
    # filter only for entries which are favorites
    dfTextAboutImages = dfTextAboutImages[dfTextAboutImages['favorite']==True][['user_id','file_name','reasons','emotions','strengths','utilization','story']]

    dfTextAboutImages['file_name'] = dfTextAboutImages['file_name'].str.replace('./', '')
    dfTextAboutImages = dfTextAboutImages.reset_index(drop=True)
    
    return dfTextAboutImages


def createImageRatings(df_images):

    # get unique image names
    imageNames = df_images['file_name'].unique()
    uniqueImageNames = []

    for imageName in imageNames:
        imageName = imageName.replace('./', '')
        uniqueImageNames.append(imageName)

    uniqueImageNames = numpy.unique(uniqueImageNames)

    # create dict for image file names and numbers
    imageNumbers = list(range(1, 148))
    imageDict = dict(zip(imageNumbers, uniqueImageNames))

    # get unique user IDs
    uniqueUsers = df_images['user_id'].unique()

    # create inverted image Dictionary
    inv_imageDict = {v: k for k, v in imageDict.items()}

    # create rating table for user per image
    ratingList = []
    for user in uniqueUsers:
        ratings=[0]*147
        imageRatingsOfUser = df_images[df_images['user_id'] == user]

        for index, row in imageRatingsOfUser.iterrows():
            key = row['file_name']
            key = key.replace('./', '')
            value = inv_imageDict[key]
            ratings[value-1] = row['rating']

        ratings.insert(0, user)
        ratingList.append(ratings)

    dfImageRatings = pd.DataFrame(ratingList)
    dfImageRatings.rename(columns={0:'user_id'}, inplace=True)
    dfImageRatings.fillna(value=0.0, inplace=True)
    
    return dfImageRatings

def createImageContents(df_images, df_imageLabels):
    
    # get important columns from images dataframe
    dfImageContents = df_images[['user_id', 'file_name', 'favorite']]
    
    # drop duplicates
    dfImageContents.drop_duplicates(inplace=True)
    
    # filter only for entries which are favorites
    dfImageContents = dfImageContents[dfImageContents['favorite']==True][['user_id','file_name']]

    # align file_name formatting
    dfImageContents['file_name'] = dfImageContents['file_name'].str.replace('./', '')

    # merge favorite images with imageLabels
    dfImageContents = dfImageContents.merge(df_imageLabels, how='inner', on='file_name')

    # drop file_name column and duplicates
    dfImageContents = dfImageContents.drop(columns=['file_name'])
    dfImageContents.drop_duplicates(inplace=True)

    # reset index
    dfImageContents = dfImageContents.reset_index(drop=True)

    return dfImageContents



# fill NaNs in dfImageDescriptions with bfill and ffill where possible, delete other NaN rows
def cleanImageDescriptions(df):
    
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


# ------------------------------------------------------------------
# -------------------------- NOT USED NOW --------------------------

def createMotives(df_mpzm):

    # get important columns from mpzm dataframe
    dfMotives = df_mpzm[['user_id','Bindung','Unternehmungslust','Macht','Geltung','Leistung']]
    
    return dfMotives


def createMood(df_mood):

    # get important columns from mood dataframe
    dfMood = df_mood[['user_id','PositiveAktivierung','NegativeAktivierung','Zufriedenheit_Glueck']]
    
    return dfMood