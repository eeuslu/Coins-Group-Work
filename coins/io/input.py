import pandas as pd
import numpy as np

def testmethod():
    print('test successful')

def get_data(datafile):
    if datafile == 'ipip':
        df = pd.read_csv('../data/input/2020-04-30_ipip_5f30f_r1.simplified_anonymized.csv', sep=';', low_memory=False)
        df = preprocess_ipip(df)
    elif datafile == 'mpzm':
        df = pd.read_csv('../data/input/2020-04-30_mpzm.simplified_anonymized.csv', sep=';', low_memory=False)
        df = preprocess_mpzm(df)
    elif datafile == 'emotions':
        df = pd.read_csv('../data/input/2020-04-30_resource_diagnostics2_emotions_anonymized.csv', sep=';', low_memory=False)
        df = preprocess_emotions(df)
    elif datafile == 'mood':
        df = pd.read_csv('../data/input/2020-04-30_mood_anonymized.csv', sep=';', low_memory=False)
        df = preprocess_mood(df)
    elif datafile == 'images':
        df = pd.read_csv('../data/input/2020-04-30_resource_diagnostics_images_anonymized.csv', sep=';', low_memory=False)
        df2 = pd.read_csv('../data/input/2020-04-30_resource_diagnostics2_images_anonymized.csv', sep=';', low_memory=False)
        df = preprocess_images(df, df2)
    elif datafile == 'sessions':
        df = pd.read_csv('../data/input/2020-04-30_resource_diagnostics_sessions_anonymized.csv', sep=';', low_memory=False)
        df2 = pd.read_csv('../data/input/2020-04-30_resource_diagnostics2_sessions_anonymized.csv', sep=';', low_memory=False)
        df = preprocess_sessions(df, df2)
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

    # # convert datatypes to boolean
    # df['job_status_edu_fulltime'] = df.job_status_edu_fulltime.astype('bool')
    # df['job_status_edu_parttime'] = df.job_status_edu_parttime.astype('bool')
    # df['job_status_employed_fulltime'] = df.job_status_employed_fulltime.astype('bool')
    # df['job_status_employed_parttime'] = df.job_status_employed_parttime.astype('bool')
    # df['job_status_selfemployed'] = df.job_status_selfemployed.astype('bool')
    # df['job_status_houskeeping'] = df.job_status_houskeeping.astype('bool')
    # df['job_status_unemployed'] = df.job_status_unemployed.astype('bool')
    # df['job_status_retired'] = df.job_status_retired.astype('bool')

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

    # # convert datatypes to boolean
    # df['job_status_edu_fulltime'] = df.job_status_edu_fulltime.astype('bool')
    # df['job_status_edu_parttime'] = df.job_status_edu_parttime.astype('bool')
    # df['job_status_employed_fulltime'] = df.job_status_employed_fulltime.astype('bool')
    # df['job_status_employed_parttime'] = df.job_status_employed_parttime.astype('bool')
    # df['job_status_selfemployed'] = df.job_status_selfemployed.astype('bool')
    # df['job_status_houskeeping'] = df.job_status_houskeeping.astype('bool')
    # df['job_status_unemployed'] = df.job_status_unemployed.astype('bool')
    # df['job_status_retired'] = df.job_status_retired.astype('bool')

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

def preprocess_emotions(df):
    df.drop_duplicates(inplace = True)
    df = df[df['reason_of_attendence'] == 'serious']
    df.drop(columns=['admin', 'accept_terms_and_conditions', 'accept_data_privacy', 'reason_of_attendence', 'created_by', 'consultant', 'quest_id'], inplace=True)
    df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    df.set_index('index', inplace=True)
    df.replace({'\n': ' '}, regex=True, inplace=True)

    #todo change datatypes

    return df

def preprocess_mood(df):
    df.drop_duplicates(inplace=True)
    df.drop(columns=['admin', 'accept_terms_and_conditions', 'accept_data_privacy', 'created_by', 'consultant'], inplace=True)
    df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    df.set_index('index', inplace=True)

    #todo change datatypes

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

    # merge two image dataframe
    # todo text formatting before merging
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

    # # convert datatypes to boolean
    # df['job_status_edu_fulltime'] = df.job_status_edu_fulltime.astype('bool')
    # df['job_status_edu_parttime'] = df.job_status_edu_parttime.astype('bool')
    # df['job_status_employed_fulltime'] = df.job_status_employed_fulltime.astype('bool')
    # df['job_status_employed_parttime'] = df.job_status_employed_parttime.astype('bool')
    # df['job_status_selfemployed'] = df.job_status_selfemployed.astype('bool')
    # df['job_status_houskeeping'] = df.job_status_houskeeping.astype('bool')
    # df['job_status_unemployed'] = df.job_status_unemployed.astype('bool')
    # df['job_status_retired'] = df.job_status_retired.astype('bool')
    # df['favorite'] = df.favorite.astype('bool')

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

    # merge two sessions dataframes
     # todo text formatting before merging
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

    # # convert datatypes to boolean
    # df['job_status_edu_fulltime'] = df.job_status_edu_fulltime.astype('bool')
    # df['job_status_edu_parttime'] = df.job_status_edu_parttime.astype('bool')
    # df['job_status_employed_fulltime'] = df.job_status_employed_fulltime.astype('bool')
    # df['job_status_employed_parttime'] = df.job_status_employed_parttime.astype('bool')
    # df['job_status_selfemployed'] = df.job_status_selfemployed.astype('bool')
    # df['job_status_houskeeping'] = df.job_status_houskeeping.astype('bool')
    # df['job_status_unemployed'] = df.job_status_unemployed.astype('bool')
    # df['job_status_retired'] = df.job_status_retired.astype('bool')

    # convert datatypes to datetime
    df['user_created_at'] = pd.to_datetime(df['user_created_at'], utc=True)
    df['last_sign_in_at'] = pd.to_datetime(df['last_sign_in_at'], utc=True)
    df['session_created_at'] = pd.to_datetime(df['session_created_at'], utc=True)

    return df