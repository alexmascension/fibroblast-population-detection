import pandas as pd
import numpy as np
import scanpy as sc


def plot_adata_cluster_properties(dict_cats_clusters={}, dict_datasets={}, what='presence', cluster_name='cluster', axis_name='axis'):
    df_cols = [i for i in list(dict_cats_clusters.keys()) if len(i) == 2]
    df_index = ['-'.join(i[:-1]) for i in dict_datasets.values()]
    df_clusters = pd.DataFrame('', index=df_index, columns=df_cols)
    
    
    for adata_name, list_adata in dict_datasets.items():
        adata = list_adata[-1]
        list_adata = list_adata[:-1]
        for cluster in df_cols:
            if what == 'presence':
                if cluster in set(adata.obs[cluster_name]):
                    df_clusters.loc['-'.join(list_adata), cluster] = '✔️'
                    
            if what == 'score':
                if cluster in set(adata.obs[cluster_name]):
                    score = adata[adata.obs[cluster_name] == cluster].obs[f'{cluster_name}_{cluster}'].values
                    df_clusters.loc['-'.join(list_adata), cluster] = f'{np.mean(score):.2f}'
                else:
                    df_clusters.loc['-'.join(list_adata), cluster] = 0
            
            if what == 'percentage':
                if cluster in set(adata.obs[cluster_name]):
                    percentage = len(adata[adata.obs[cluster_name] == cluster])/len(adata)
                    df_clusters.loc['-'.join(list_adata), cluster] = f'{percentage:.2f}'
                else:
                    df_clusters.loc['-'.join(list_adata), cluster] = 0
                    
            if what == 'axis':
                if cluster in set(adata.obs[cluster_name]):
                    axis_val = adata[adata.obs[cluster_name] == cluster].obs[axis_name].values[0]
                    df_clusters.loc['-'.join(list_adata), cluster] = axis_val
                    
    return df_clusters


def clear_adata(adata): 
    columns = adata.obs.columns
    del_cols = [i for i in columns if ('cluster_' in i) | ('assigned_cats_' in i) | ('axis_' in i) ]
    for col in del_cols:
        del adata.obs[col]
    
    return adata