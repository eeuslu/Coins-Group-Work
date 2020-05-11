import pandas as pd
import numpy as numpy

def createPersonality(df_ipip):

    dfPersonality = df_ipip[['user_id','N','E','O','A','C']]
    dfPersonality.columns=['user_id','neurotizismus','extraversion','offenheit','vertraeglichkeit','gewissenhaftigkeit']

    dfPersonality.drop_duplicates(inplace=True)
    return dfPersonality


def createSocialDemographics(df_ipip, df_mpzm, df_images, df_emotions, df_mood): #df_sessions --needed?
    
    sources = [df_mpzm, df_images, df_emotions, df_mood]
    #Create first DataFrame as base for concat
    dfSocialDemographics = df_ipip[['user_id','gender','registration_age','registration_ageKat','country','work_country','work_district',
                            'job_position','job_sector','company_size','job_status_edu_parttime','job_status_edu_fulltime',
                            'job_status_employed_parttime','job_status_employed_fulltime','job_status_selfemployed',
                            'job_status_houskeeping','job_status_unemployed','job_status_retired','educational_achievement']]

    for source in sources:
        dfSocialDemographicsSource = source[['user_id','gender','registration_age','registration_ageKat','country','work_country','work_district',
                            'job_position','job_sector','company_size','job_status_edu_parttime','job_status_edu_fulltime',
                            'job_status_employed_parttime','job_status_employed_fulltime','job_status_selfemployed',
                            'job_status_houskeeping','job_status_unemployed','job_status_retired','educational_achievement']]
        dfSocialDemographics = pd.concat([dfSocialDemographics,dfSocialDemographicsSource],ignore_index=True)
    dfSocialDemographics.drop_duplicates(inplace=True)

    return dfSocialDemographics
    

#RENAME !! Don'tknopw the wanted name
def textAboutImages(df_images):

    dfTextAboutImages = df_images[['user_id','file_name','reasons','emotions','strengths','utilization','story','favorite']]
    dfTextAboutImages.drop_duplicates(inplace=True)
    
    return dfTextAboutImages[dfTextAboutImages['favorite']==True][['user_id','file_name','reasons','emotions','strengths','utilization','story']]


#RENAME !! Don'tknopw the wanted name
def traids(df_mpzm):

    dfTraids = df_mpzm[['user_id','Bindung','Unternehmungslust','Macht','Geltung','Leistung']]
    return dfTraids



def happiness(df_mood):

    dfMood = df_mood[['user_id','PositiveAktivierung','NegativeAktivierung','Zufriedenheit_Glueck']]
    return dfMood



#NOT FUNCTIONAL
def imageRating(df_images):

    #Get unique Names
    imageNames = df_images['file_name'].unique()

    #Group by user
    groups = df_images.groupby('user_id')

    dfImageRating = pd.DataFrame(None,columns=['user_id'])

    #Create Df
    for i,imageName in enumerate(imageNames):
        naming = "Image_{number}"
        dfImageRating['naming'.format(number=str(i))]

    
    #Collect Data
    data = []
    l = [None] * len(imageNames)

    for group in groups:
        #get index in filename
        imageNames.get_index

