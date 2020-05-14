import pandas as pd
import numpy as numpy

def createPersonality(df_ipip):

    # get important columns from ipip dataframe
    dfPersonality = df_ipip[['user_id','N','E','O','A','C']]
    dfPersonality.columns=['user_id','neurotizismus','extraversion','offenheit','vertraeglichkeit','gewissenhaftigkeit']

    #drop duplicates
    dfPersonality.drop_duplicates(inplace=True)

    return dfPersonality


def createSocialDemographics(df_ipip, df_mpzm, df_images, df_emotions, df_mood):
    
    sources = [df_mpzm, df_images, df_emotions, df_mood]
    
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