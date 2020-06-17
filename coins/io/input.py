import pandas as pd
import numpy as np
from .utils import get_data_path
import os
import yaml
from sklearn.externals.joblib import dump, load


# return API credentials from the local credentials.yaml file
def getAPIcredentials(credential):
    path = os.path.join(get_data_path(),'input/credentials.yaml')
    with open(path, 'r') as credentials:
        credentials = yaml.load(credentials)
    
    if credential == 'apiKey_ibmWatson':
        apiKey_ibmWatson = credentials['ibmWatson']['apiKey']
        return apiKey_ibmWatson
    elif credential == 'apiURL_ibmWatson':
        apiURL_ibmWatson = credentials['ibmWatson']['apiURL']
        return apiURL_ibmWatson
    elif credential == 'apiKey_deepL':
        apiKey_deepL = credentials['deepL']['apiKey']
        return apiKey_deepL
    else:
        return 'the credential you are looking for was not found.'


def loadInitialDFs(datafile):
    if datafile == 'personality':
        path = os.path.join(get_data_path(),'output/initialDataFrames/personality.csv')
        df = pd.read_csv(path, sep=';', decimal=',', low_memory=False)
        df = df.drop(columns=['Unnamed: 0'])
    elif datafile == 'imageRatings':
        path = os.path.join(get_data_path(),'output/initialDataFrames/imageRatings.csv')
        df = pd.read_csv(path, sep=';', decimal=',', low_memory=False)
        df = df.drop(columns=['Unnamed: 0'])
    elif datafile == 'imageDescriptions':
        path = os.path.join(get_data_path(),'output/initialDataFrames/imageDescriptions.csv')
        df = pd.read_csv(path, sep=';', decimal=',', low_memory=False)
        df = df.drop(columns=['Unnamed: 0'])
    elif datafile == 'socioDemographics':
        path = os.path.join(get_data_path(),'output/initialDataFrames/socioDemographics.csv')
        df = pd.read_csv(path, sep=';', decimal=',', low_memory=False)
        df = df.drop(columns=['Unnamed: 0'])
    elif datafile == 'imageContents':
        path = os.path.join(get_data_path(),'output/initialDataFrames/imageContents.csv')
        df = pd.read_csv(path, sep=';', decimal=',', low_memory=False)
        df = df.drop(columns=['Unnamed: 0'])
    return df


def loadAnalyzedImageDescriptions():
    path = os.path.join(get_data_path(),'output/analyzedDataFrames/analyzedImageDescriptions.csv')
    df = pd.read_csv(path, sep=';', decimal=',', low_memory=False)
    df = df.drop(columns=['Unnamed: 0'])
    return df

def loadPreparedDFs(datafile):
    if datafile == 'personality':
        path = os.path.join(get_data_path(),'output/preparedDataFrames/personality.csv')
        df = pd.read_csv(path, sep=';', decimal=',', low_memory=False)
        df = df.drop(columns=['Unnamed: 0'])
    elif datafile == 'imageRatings':
        path = os.path.join(get_data_path(),'output/preparedDataFrames/imageRatings.csv')
        df = pd.read_csv(path, sep=';', decimal=',', low_memory=False)
        df = df.drop(columns=['Unnamed: 0'])
    elif datafile == 'imageDescriptions':
        path = os.path.join(get_data_path(),'output/preparedDataFrames/imageDescriptions.csv')
        df = pd.read_csv(path, sep=';', decimal=',', low_memory=False)
        df = df.drop(columns=['Unnamed: 0'])
    elif datafile == 'socioDemographics':
        path = os.path.join(get_data_path(),'output/preparedDataFrames/socioDemographics.csv')
        df = pd.read_csv(path, sep=';', decimal=',', low_memory=False)
        df = df.drop(columns=['Unnamed: 0'])
    elif datafile == 'imageContents':
        path = os.path.join(get_data_path(),'output/preparedDataFrames/imageContents.csv')
        df = pd.read_csv(path, sep=';', decimal=',', low_memory=False)
        df = df.drop(columns=['Unnamed: 0'])
    return df


def getPreprocessedRawData(datafile):
    if datafile == 'ipip':
        path = os.path.join(get_data_path(),'input/ipip.csv')
        df = pd.read_csv(path, sep=';', low_memory=False)
        df = preprocess_ipip(df)
    elif datafile == 'mpzm':
        path = os.path.join(get_data_path(),'input/mpzm.csv')
        df = pd.read_csv(path, sep=';', low_memory=False)
        df = preprocess_mpzm(df)
    elif datafile == 'mood':
        path = os.path.join(get_data_path(),'input/mood.csv')
        df = pd.read_csv(path, sep=';', low_memory=False)
        df = preprocess_mood(df)
    elif datafile == 'images':
        path1 = os.path.join(get_data_path(),'input/images_v1.csv')
        path2 = os.path.join(get_data_path(),'input/images_v2.csv')
        df = pd.read_csv(path1, sep=';', low_memory=False)
        df2 = pd.read_csv(path2, sep=';', low_memory=False)
        df = preprocess_images(df, df2)
    elif datafile == 'sessions':
        path1 = os.path.join(get_data_path(),'input/sessions_v1.csv')
        path2 = os.path.join(get_data_path(),'input/sessions_v2.csv')
        df = pd.read_csv(path1, sep=';', low_memory=False)
        df2 = pd.read_csv(path2, sep=';', low_memory=False)
        df = preprocess_sessions(df, df2)
    elif datafile == 'imageLabels':
        path = os.path.join(get_data_path(),'input/imageLabels.csv')
        df = pd.read_csv(path, sep=';', low_memory=False)
    return df


def preprocess_ipip(df):
    df.drop_duplicates(inplace = True)
    df = df[df['reason_of_attendance'] == 'serious']
    df.drop(columns=['admin', 'accept_terms_and_conditions', 'accept_data_privacy', 'reason_of_attendance', 'created_by', 'consultant', 'quest_id', 'coordinates', 'time_zone', 'sequence'], inplace=True)
    df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    df.set_index('index', inplace=True)

    # convert datatypes to category
    df['gender'] = df.gender.astype('category')
    df['country'] = df.country.astype('category')
    df['work_country'] = df.work_country.astype('category')
    df['work_district'] = df.work_district.astype('category')
    df['job_position'] = df.job_position.astype('category')
    df['job_sector'] = df.job_sector.astype('category')
    df['found_over'] = df.found_over.astype('category')
    df['company_size'] = df.company_size.astype('category')
    df['educational_achievement'] = df.educational_achievement.astype('category')
    df['registration_ageKat'] = df.registration_ageKat.astype('category')
    df['version'] = df.version.astype('category')
    df['ageKat'] = df.ageKat.astype('category')

    # convert datatypes to datetime
    df['user_created_at'] = pd.to_datetime(df['user_created_at'], utc=True)
    df['last_sign_in_at'] = pd.to_datetime(df['last_sign_in_at'], utc=True)
    df['session_created_at'] = pd.to_datetime(df['session_created_at'], utc=True)
    df['session_updated_at'] = pd.to_datetime(df['session_updated_at'], utc=True)

    # convert datatypes to float64
    df['N'] = df.N.str.replace(',','.').astype('float64')
    df['E'] = df.E.str.replace(',','.').astype('float64')
    df['O'] = df.O.str.replace(',','.').astype('float64')
    df['A'] = df.A.str.replace(',','.').astype('float64')
    df['C'] = df.C.str.replace(',','.').astype('float64')
    df['N_aengstlichkeit'] = df.N_aengstlichkeit.str.replace(',','.').astype('float64')
    df['N_reizbarkeit'] = df.N_reizbarkeit.str.replace(',','.').astype('float64')
    df['N_depression'] = df.N_depression.str.replace(',','.').astype('float64')
    df['N_befangenheit'] = df.N_befangenheit.str.replace(',','.').astype('float64')
    df['N_impulsivitaet'] = df.N_impulsivitaet.str.replace(',','.').astype('float64')
    df['N_verletzlichkeit'] = df.N_verletzlichkeit.str.replace(',','.').astype('float64')
    df['E_herzlichkeit'] = df.E_herzlichkeit.str.replace(',','.').astype('float64')
    df['E_geselligkeit'] = df.E_geselligkeit.str.replace(',','.').astype('float64')
    df['E_durchsetzungsvermoegen'] = df.E_durchsetzungsvermoegen.str.replace(',','.').astype('float64')
    df['E_aktivitaet'] = df.E_aktivitaet.str.replace(',','.').astype('float64')
    df['E_erlebnishunger'] = df.E_erlebnishunger.str.replace(',','.').astype('float64')
    df['E_frohsinn'] = df.E_frohsinn.str.replace(',','.').astype('float64')
    df['O_phantasie'] = df.O_phantasie.str.replace(',','.').astype('float64')
    df['O_aesthetik'] = df.O_aesthetik.str.replace(',','.').astype('float64')
    df['O_gefuehle'] = df.O_gefuehle.str.replace(',','.').astype('float64')
    df['O_handlungen'] = df.O_handlungen.str.replace(',','.').astype('float64')
    df['O_ideen'] = df.O_ideen.str.replace(',','.').astype('float64')
    df['O_werte'] = df.O_werte.str.replace(',','.').astype('float64')
    df['A_vertrauen'] = df.A_vertrauen.str.replace(',','.').astype('float64')
    df['A_freimuetigkeit'] = df.A_freimuetigkeit.str.replace(',','.').astype('float64')
    df['A_altruismus'] = df.A_altruismus.str.replace(',','.').astype('float64')
    df['A_entgegenkommen'] = df.A_entgegenkommen.str.replace(',','.').astype('float64')
    df['A_bescheidenheit'] = df.A_bescheidenheit.str.replace(',','.').astype('float64')
    df['A_gutherzigkeit'] = df.A_gutherzigkeit.str.replace(',','.').astype('float64')
    df['C_kompetenz'] = df.C_kompetenz.str.replace(',','.').astype('float64')
    df['C_ordnungsliebe'] = df.C_ordnungsliebe.str.replace(',','.').astype('float64')
    df['C_pflichtbewusstsein'] = df.C_pflichtbewusstsein.str.replace(',','.').astype('float64')
    df['C_leistungsstreben'] = df.C_leistungsstreben.str.replace(',','.').astype('float64')
    df['C_selbstdisziplin'] = df.C_selbstdisziplin.str.replace(',','.').astype('float64')
    df['C_besonnenheit'] = df.C_besonnenheit.str.replace(',','.').astype('float64')

    return df

def preprocess_mpzm(df):
    df.drop_duplicates(inplace = True)
    df = df[df['reason_of_attendance'] == 'serious']
    df.drop(columns=['admin', 'accept_terms_and_conditions', 'accept_data_privacy', 'reason_of_attendance', 'created_by', 'consultant', 'quest_id', 'coordinates', 'time_zone', 'sequence'], inplace=True)
    df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    df.set_index('index', inplace=True)

    # convert datatypes to category
    df['gender'] = df.gender.astype('category')
    df['country'] = df.country.astype('category')
    df['work_country'] = df.work_country.astype('category')
    df['work_district'] = df.work_district.astype('category')
    df['job_position'] = df.job_position.astype('category')
    df['job_sector'] = df.job_sector.astype('category')
    df['found_over'] = df.found_over.astype('category')
    df['company_size'] = df.company_size.astype('category')
    df['educational_achievement'] = df.educational_achievement.astype('category')
    df['registration_ageKat'] = df.registration_ageKat.astype('category')
    df['version'] = df.version.astype('category')
    df['ageKat'] = df.ageKat.astype('category')

    # convert datatypes to float64
    df['Bindung'] = df.Bindung.str.replace(',','.').astype('float64')
    df['Unternehmungslust'] = df.Unternehmungslust.str.replace(',','.').astype('float64')
    df['Macht'] = df.Macht.str.replace(',','.').astype('float64')
    df['Geltung'] = df.Geltung.str.replace(',','.').astype('float64')
    df['Leistung'] = df.Leistung.str.replace(',','.').astype('float64')

    # fill missing datetimes by logical equivalent (sign_in_count == 0)
    df.last_sign_in_at.replace('                       ', np.nan, inplace=True)
    df.last_sign_in_at.fillna(df.user_created_at, inplace=True)

    # convert datatypes to datetime
    df['user_created_at'] = pd.to_datetime(df['user_created_at'], utc=True)
    df['last_sign_in_at'] = pd.to_datetime(df['last_sign_in_at'], utc=True)
    df['session_created_at'] = pd.to_datetime(df['session_created_at'], utc=True)
    df['session_updated_at'] = pd.to_datetime(df['session_updated_at'], utc=True)

    return df

def preprocess_mood(df):
    df.drop_duplicates(inplace=True)
    df.drop(columns=['admin', 'accept_terms_and_conditions', 'accept_data_privacy', 'created_by', 'consultant', 'coordinates', 'time_zone'], inplace=True)
    df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    df.set_index('index', inplace=True)

    # convert datatypes to category
    df['gender'] = df.gender.astype('category')
    df['country'] = df.country.astype('category')
    df['work_country'] = df.work_country.astype('category')
    df['work_district'] = df.work_district.astype('category')
    df['job_position'] = df.job_position.astype('category')
    df['job_sector'] = df.job_sector.astype('category')
    df['found_over'] = df.found_over.astype('category')
    df['company_size'] = df.company_size.astype('category')
    df['educational_achievement'] = df.educational_achievement.astype('category')
    df['registration_ageKat'] = df.registration_ageKat.astype('category')
    df['version'] = df.version.astype('category')
    df['ageKat'] = df.ageKat.astype('category')

    # fill missing datetimes by logical equivalent (sign_in_count == 0)
    df.last_sign_in_at.replace('                       ', np.nan, inplace=True)
    df.last_sign_in_at.fillna(df.user_created_at, inplace=True)

    # convert datatypes to datetime
    df['user_created_at'] = pd.to_datetime(df['user_created_at'], utc=True)
    df['last_sign_in_at'] = pd.to_datetime(df['last_sign_in_at'], utc=True)
    df['session_created_at'] = pd.to_datetime(df['session_created_at'], utc=True)
    df['session_updated_at'] = pd.to_datetime(df['session_updated_at'], utc=True)

    # convert datatypes to float64
    df['PositiveAktivierung'] = df.PositiveAktivierung.str.replace(',','.').astype('float64')
    df['NegativeAktivierung'] = df.NegativeAktivierung.str.replace(',','.').astype('float64')
    df['Zufriedenheit_Glueck'] = df.Zufriedenheit_Glueck.str.replace(',','.').astype('float64')

    return df

def preprocess_images(df, df2):
    df.drop_duplicates(inplace = True)
    df2.drop_duplicates(inplace = True)
    df = df[df['reason_of_attendance'] == 'serious']
    df2 = df2[df2['reason_of_attendence'] == 'serious']
    df.drop(columns=['admin', 'accept_terms_and_conditions', 'accept_data_privacy', 'reason_of_attendance', 'created_by', 'consultant', 'quest_id', 'image_created_at', 'displays_humans'], inplace=True)
    df2.drop(columns=['admin', 'accept_terms_and_conditions', 'accept_data_privacy', 'reason_of_attendence', 'created_by', 'consultant', 'quest_id', 'completed_at'], inplace=True)
    df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    df2.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    df.set_index('index', inplace=True)
    df2.set_index('index', inplace=True)
    df.replace({'\n': ' '}, regex=True, inplace=True)
    df2.replace({'\n': ' '}, regex=True, inplace=True)

    # merge the two different image dataframes
    df['emotion_recognition_enabled'] = False
    df['category_id'] = np.nan
    df2.rename(columns={'name': 'file_name', 'id': 'image_id'}, inplace=True)
    df.rename(columns={'sorted_tags': 'categories', 'tag': 'category_name'}, inplace=True)
    df = df.append(df2, sort=False)
    df = df.reset_index(drop=True)

    # convert datatypes to category
    df['gender'] = df.gender.astype('category')
    df['country'] = df.country.astype('category')
    df['work_country'] = df.work_country.astype('category')
    df['work_district'] = df.work_district.astype('category')
    df['job_position'] = df.job_position.astype('category')
    df['job_sector'] = df.job_sector.astype('category')
    df['found_over'] = df.found_over.astype('category')
    df['company_size'] = df.company_size.astype('category')
    df['educational_achievement'] = df.educational_achievement.astype('category')
    df['registration_ageKat'] = df.registration_ageKat.astype('category')
    df['version'] = df.version.astype('category')

    # convert datatypes to datetime
    df['user_created_at'] = pd.to_datetime(df['user_created_at'], utc=True)
    df['last_sign_in_at'] = pd.to_datetime(df['last_sign_in_at'], utc=True)
    df['session_created_at'] = pd.to_datetime(df['session_created_at'], utc=True)
    df['shown_at'] = pd.to_datetime(df['shown_at'], utc=True)
    df['rated_at'] = pd.to_datetime(df['rated_at'], utc=True)

    return df

def preprocess_sessions(df, df2):
    df.drop_duplicates(inplace = True)
    df2.drop_duplicates(inplace = True)
    df = df[df['reason_of_attendance'] == 'serious']
    df2 = df2[df2['reason_of_attendence'] == 'serious']
    df.drop(columns=['admin', 'accept_terms_and_conditions', 'accept_data_privacy', 'reason_of_attendance', 'created_by', 'consultant', 'quest_id'], inplace=True)
    df2.drop(columns=['admin', 'accept_terms_and_conditions', 'accept_data_privacy', 'reason_of_attendence', 'created_by', 'consultant', 'quest_id', 'completed_at'], inplace=True)
    df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    df2.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    df.set_index('index', inplace=True)
    df2.set_index('index', inplace=True)
    df.replace({'\n': ' '}, regex=True, inplace=True)
    df2.replace({'\n': ' '}, regex=True, inplace=True)

    # merge the two different session dataframes
    df['emotion_recognition_enabled'] = False
    df.rename(columns={'sorted_tags': 'categories'}, inplace=True)
    df = df.append(df2, sort=False)
    df = df.reset_index(drop=True)

    # convert datatypes to category
    df['gender'] = df.gender.astype('category')
    df['country'] = df.country.astype('category')
    df['work_country'] = df.work_country.astype('category')
    df['work_district'] = df.work_district.astype('category')
    df['job_position'] = df.job_position.astype('category')
    df['job_sector'] = df.job_sector.astype('category')
    df['found_over'] = df.found_over.astype('category')
    df['company_size'] = df.company_size.astype('category')
    df['educational_achievement'] = df.educational_achievement.astype('category')
    df['registration_ageKat'] = df.registration_ageKat.astype('category')
    df['version'] = df.version.astype('category')

    # convert datatypes to datetime
    df['user_created_at'] = pd.to_datetime(df['user_created_at'], utc=True)
    df['last_sign_in_at'] = pd.to_datetime(df['last_sign_in_at'], utc=True)
    df['session_created_at'] = pd.to_datetime(df['session_created_at'], utc=True)

    return df

def loadBestResult(targetDataFrameName):
    path = os.path.join(get_data_path(), 'output/modelResults/' + targetDataFrameName + '/bestResults.csv')
    df = pd.read_csv(path, sep=';', decimal=',', low_memory=False)
    return df

def loadModel(targetFeatureName, targetDataFrameName):
    targetFeatureName = targetFeatureName.replace("/", "")
    #Save the model
    path = os.path.join(get_data_path(), 'output/modelResults/' + targetDataFrameName + '/model/' + targetFeatureName + 'Model.pkl')
    model = load(path)

    #Save the PCA
    path = os.path.join(get_data_path(), 'output/modelResults/' + targetDataFrameName + '/pca/' + targetFeatureName + 'PCA.pkl')
    pca = load(path)

    #Save the StandardScaler
    path = os.path.join(get_data_path(), 'output/modelResults/' + targetDataFrameName + '/standardScaler/' + targetFeatureName + 'StandardScaler.pkl')
    standardScaler = load(path)

    return model, pca, standardScaler