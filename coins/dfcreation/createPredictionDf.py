import pandas as pd
import numpy as np
from scipy import stats
import itertools
from coins.io.input import loadBestResult, loadModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import coins.io.input as input
import coins.io.output as output
import coins.nluTranslation.sentimentAnalysis as sentimentAnalysis
import coins.nluTranslation.translation as translation
import coins.dfcreation.dfcreation as dfcreation
import coins.correlation.prepareDataframes as prepareDataframes
import coins.correlation

def preprocessImageContent(images, imageLabels):
    images.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    images.set_index('index', inplace=True)
    # get relevant columns
    images = images[['user_id','favirte_image']]
    images.rename(columns={'favirte_image': 'file_name'}, inplace=True)
    images['file_name'] = images['file_name'].str.replace('./', '')
    # merge favorite images with imageLabels
    dfImageContents = images.merge(imageLabels, how='inner', on='file_name')
    # drop file_name column and duplicates
    dfImageContents = dfImageContents.drop(columns=['file_name'])
    dfImageContents.drop_duplicates(inplace=True)
    # reset index
    dfImageContents = dfImageContents.reset_index(drop=True)
    dfImageContents.dropna(inplace=True)
    return dfImageContents

def preprocessImageDescriptions(images):
    # get important columns from images dataframe
    dfImageDescriptions = images[['user_id','reasons','emotions','strengths','utilization','story']]
    # Analyze and save imageDescriptions (needs credentials, costs money)
    dfImageDescriptions = translation.translateToEnglish(dfImageDescriptions)
    dfImageDescriptions = sentimentAnalysis.analyzeEnglishSentimentAndEmotions(dfImageDescriptions)
    dfImageDescriptions = sentimentAnalysis.fillImageDescriptions(dfImageDescriptions)
    # save results
    output.saveAnalyzedImageDescriptions(dfImageDescriptions)

    dfImageDescriptions = prepareDataframes.prepareImageDescriptions(dfImageDescriptions, multiclass=False, train=False, split='median')
    return dfImageDescriptions

def preprocessImageRatings(images):
    images.rename(columns={'name': 'file_name'}, inplace=True)
    dfImageRatings = dfcreation.createImageRatings(images, train=False)
    return dfImageRatings

def preprocessPersonality(ipip):
    # get important columns from images dataframe
    dfPersonality = ipip[['user_id','N','E','O','A','C']]
    dfPersonality.rename(columns={"N": "neurotizismus", "E": "extraversion", "O": "offenheit", "A": "vertraeglichkeit", "C": "gewissenhaftigkeit"}, inplace=True)
    dfPersonality = prepareDataframes.preparePersonality(dfPersonality, multiclass=False, split='mean', train=False)
    return dfPersonality

def preprocessSocioDemographics(socioDem):
    # get important columns from dataframe
    dfSocioDemographics = socioDem[['user_id','gender','country','work_country','work_district','job_position', 'job_sector', 'company_size', 'job_status_edu_fulltime', 'job_status_edu_parttime', 'job_status_employed_fulltime', 'job_status_employed_parttime', 'job_status_selfemployed', 'job_status_houskeeping', 'job_status_unemployed', 'job_status_retired', 'educational_achievement', 'registration_ageKat']]

     # convert datatypes to category
    dfSocioDemographics['gender'] = dfSocioDemographics.gender.astype('category')
    dfSocioDemographics['country'] = dfSocioDemographics.country.astype('category')
    dfSocioDemographics['work_country'] = dfSocioDemographics.work_country.astype('category')
    dfSocioDemographics['work_district'] = dfSocioDemographics.work_district.astype('category')
    dfSocioDemographics['job_position'] = dfSocioDemographics.job_position.astype('category')
    dfSocioDemographics['job_sector'] = dfSocioDemographics.job_sector.astype('category')
    dfSocioDemographics['company_size'] = dfSocioDemographics.company_size.astype('category')
    dfSocioDemographics['educational_achievement'] = dfSocioDemographics.educational_achievement.astype('category')
    dfSocioDemographics['registration_ageKat'] = dfSocioDemographics.registration_ageKat.astype('category')
    
    dfSocioDemographics, _ = coins.correlation.prepareSocioDemographics(dfSocioDemographics, 1)

    dfSocioDemographics.rename({'company_size_1': 'company_size_1.0', 'company_size_249': 'company_size_249.0', 'company_size_250': 'company_size_250.0', 'company_size_49': 'company_size_49.0', 'company_size_9': 'company_size_9.0',}, inplace=True, axis=1)

    socDemStructure = input.loadDf("input/SystemOwned/columnsSocioDemographics.csv", header=True)
    for column in socDemStructure.columns:
        if column not in dfSocioDemographics:
            dfSocioDemographics[column] = 0

    return dfSocioDemographics