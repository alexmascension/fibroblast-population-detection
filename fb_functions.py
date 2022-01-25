import pandas as pd
import numpy as np
import scanpy as sc


def plot_adata_cluster_properties(dict_cats_clusters={}, list_datasets=[], what='presence', cluster_name='cluster', axis_name='axis'):
    df_cols = [i for i in list(dict_cats_clusters.keys()) if len(i) == 2]
    df_index = []
    for adata in list_datasets:
        df_index.append(f"{adata.obs['Author'].values[0]}-{int(adata.obs['Year'].values[0])}-{'/'.join(adata.obs['Condition'].cat.categories)}")
    
    df_clusters = pd.DataFrame('', index=df_index, columns=df_cols)
    
    
    for adata, name_dataset in zip(list_datasets, df_index):
        for cluster in df_cols:
            if what == 'presence':
                if cluster in set(adata.obs[cluster_name]):
                    df_clusters.loc[name_dataset, cluster] = '✔️'
                    
            if what == 'score':
                if cluster in set(adata.obs[cluster_name]):
                    score = adata[adata.obs[cluster_name] == cluster].obs[f'{cluster_name}_{cluster}'].values
                    df_clusters.loc[name_dataset, cluster] = f'{np.mean(score):.2f}'
                else:
                    df_clusters.loc[name_dataset, cluster] = 0
            
            if what == 'percentage':
                if cluster in set(adata.obs[cluster_name]):
                    percentage = len(adata[adata.obs[cluster_name] == cluster])/len(adata)
                    df_clusters.loc[name_dataset, cluster] = f'{percentage:.2f}'
                else:
                    df_clusters.loc[name_dataset, cluster] = 0
                    
            if what == 'axis':
                if cluster in set(adata.obs[cluster_name]):
                    axis_val = adata[adata.obs[cluster_name] == cluster].obs[axis_name].values[0]
                    df_clusters.loc[name_dataset, cluster] = axis_val
                    
    return df_clusters


def clear_adata(adata): 
    columns = adata.obs.columns
    del_cols = [i for i in columns if ('cluster_' in i) | ('assigned_cats_' in i) | ('axis_' in i) ]
    for col in del_cols:
        del adata.obs[col]
    
    return adata