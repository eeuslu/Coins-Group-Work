import pandas as pd
from sklearn.cluster import KMeans


def cluster(dfInput,cluster=5, onlyCluster=False):
    
    dfWithId = dfInput.copy()

    if cluster > 0:
        dfWithoutId = dfWithId.drop('user_id',axis=1)

        kmeans = KMeans(n_clusters=cluster, random_state=0).fit(dfWithoutId)

        prediction = kmeans.predict(dfWithoutId)
        dfWithId['cluster'] = prediction
        dfWithId['cluster'] = dfWithId['cluster'].astype('float')

        if onlyCluster == True:
            for column in dfWithId.columns:
                if column not in ['user_id','cluster']:
                    dfWithId.drop(column,axis=1,inplace=True)

    return dfWithId


