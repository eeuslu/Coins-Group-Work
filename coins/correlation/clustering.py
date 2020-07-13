import coins
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import time
import itertools

from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns

################################################################################################################################################################

#Cluster a dataframe to the overgiven number of clusters
#If onlyCluster == True returns only cluster number and user_id
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



#Returns the loss of clusters with different k
#Allows to choose a good k manually
def find_Good_K(dfInput, kMax=15):

    dfWithId = dfInput.copy()
    dfWithoutId = dfWithId.drop('user_id',axis=1)

    # clustering with k-means
    clusters = []
    losses = []

    for i in range(kMax):
        model = KMeans(n_clusters=i+1, n_jobs=-1)
        model.fit(dfWithoutId)
        clusters.append(i+1)
        losses.append(model.inertia_)

    plt.plot(clusters, losses)
    plt.xlim([0,15])
    plt.xlabel("Number of Cluster")
    plt.ylabel("Loss Value")
    plt.title("Figure 21: K-Means Loss for Trip Types")
    plt.show()


