import pandas as pd
import numpy as np
import requests

# translate one specified column of dfImageDescriptions from german to english and append it to the DataFrame, using DeepL API
def translateToEnglish(df, columnToTranslate):

    url = 'https://api.deepl.com/v2/translate'
    deepLAuthKey = 'place API Key here'
    sourceLang = 'DE'
    targetLang = 'EN'
    
    # empty list for translated strings
    translation = []

    # iterate through every row of the dataframe and ask for sentiment score if cell is not NaN, save sentiment in list
    for index, row in df.iterrows():
        if pd.isnull(row[columnToTranslate]):
            translation.append(np.NaN)
        elif row[columnToTranslate] == ' ':
            translation.append(np.NaN)
        else:
            textToTranslate = str(row[columnToTranslate])
            payload = {'auth_key': deepLAuthKey, 'text': textToTranslate, 'source_lang': sourceLang, 'target_lang': targetLang}
            response = requests.request("POST", url=url, data=payload)
            translatedText = response.json()['translations'][0]['text']
            translation.append(translatedText)

    # add all sentiment scores as column to dataframe
    df[columnToTranslate + '_translation'] = translation

    # replace empty cells with NaN
    df = df.replace(' ', np.NaN)

    # return dataframe
    return df