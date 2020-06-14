import pandas as pd
import os
from .utils import get_data_path

def saveInitialDFs(df, fileName):
    path = os.path.join(get_data_path(),'output/initialDataFrames/' + fileName + '.csv')
    df.to_csv(path, sep=';', decimal=',', encoding='utf-8')

def savePreparedImageDescriptions(df):
    path = os.path.join(get_data_path(),'output/initialDataFrames/preparedImageDescriptions.csv')
    df.to_csv(path, sep=';', decimal=',', encoding='utf-8')