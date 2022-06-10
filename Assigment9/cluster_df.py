import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage

def cluster_df(df, method='single', threshold=100):
    '''
    Accepts a square distance matrix as an indexed DataFrame and returns a dict of index keyed flat clusters 
    Performs single linkage clustering by default, see scipy.cluster.hierarchy.linkage docs for others
    '''
   
    dm_cnd = squareform(df.values)
    clusters = fcluster(linkage(dm_cnd,
                                method=method,
                                metric='precomputed'),
                        criterion='distance',
                        t=threshold)
    names_clusters = {s:c for s, c in zip(df.columns, clusters)}
    return names_clusters