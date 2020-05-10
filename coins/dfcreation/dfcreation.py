import pandas as pd
import numpy as numpy

def createPersonality(df_ipip):

    dfPersonality = df_ipip[['user_id','N','E','O','A','C']]
    dfPersonality.columns=['user_id','neurotizismus','extraversion','offenheit','vertraeglichkeit','gewissenhaftigkeit']

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
        dfSocialDemographics = pd.concat([dfSocialDemographics,dfSocialDemographicsSource], axis=1)

    return dfSocialDemographics
    
