import pandas as pd
import numpy as numpy

def createPersonality(df_ipip):
    dfPersonality = pd.DataFrame()
    dfPersonality['user_id'] = df_ipip['user_id']
    dfPersonality['neuroticism'] = df_ipip['N']
    dfPersonality['extraversion'] = df_ipip['E']
    dfPersonality['openness'] = df_ipip['O']
    dfPersonality['agreeableness'] = df_ipip['A']
    dfPersonality['conscience'] = df_ipip['C']

    return dfPersonality

def createSocialDemographics(df_ipip, df_mpzm, df_images, df_sessions, df_emotions, df_mood):
    dfSocialDemographics = pd.DataFrame()
    dfSocialDemographics['user_id'] = df_ipip['user_id']
    dfSocialDemographics['gender'] = df_ipip['gender']
    dfSocialDemographics['age'] = df_ipip['age']
    dfSocialDemographics['ageKat'] = df_ipip['ageKat']
    dfSocialDemographics['country'] = df_ipip['country']
    dfSocialDemographics['work_country'] = df_ipip['work_country']
    dfSocialDemographics['work_district'] = df_ipip['work_district']
    dfSocialDemographics['job_position'] = df_ipip['job_position']
    dfSocialDemographics['job_sector'] = df_ipip['job_sector']
    dfSocialDemographics['company_size'] = df_ipip['company_size']
    dfSocialDemographics['job_status_edu_parttime'] = df_ipip['job_status_edu_parttime']
    dfSocialDemographics['job_status_edu_fulltime'] = df_ipip['job_status_edu_fulltime']
    dfSocialDemographics['job_status_employed_parttime'] = df_ipip['job_status_employed_parttime']
    dfSocialDemographics['job_status_employed_fulltime'] = df_ipip['job_status_employed_fulltime']
    dfSocialDemographics['job_status_selfemployed'] = df_ipip['job_status_selfemployed']
    dfSocialDemographics['job_status_houskeeping'] = df_ipip['job_status_houskeeping']
    dfSocialDemographics['job_status_unemployed'] = df_ipip['job_status_unemployed']
    dfSocialDemographics['job_status_retired'] = df_ipip['job_status_retired']
    dfSocialDemographics['educational_achievement'] = df_ipip['educational_achievement']

    #TODO Merging with other all other dataframes

    return dfSocialDemographics
    