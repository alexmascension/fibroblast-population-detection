import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as spr
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import matplotlib.cm as cm


df_metadata = pd.read_csv('data/sample_metadata.csv', sep='\t')

magma = [plt.get_cmap('magma')(i) for i in np.linspace(0,1, 80)]
magma[0] = (0.88, 0.88, 0.88, 1)
magma = mpl.colors.LinearSegmentedColormap.from_list("", magma[:65])

def metadata_assignment(adata, author, year, batch, do_return=False, do_sparse=True):
    adata.obs[df_metadata.columns] = np.tile(df_metadata[(df_metadata.Author == author) & (df_metadata.Year == year) & 
                                                                 (df_metadata['Internal sample identifier'] == batch)].values, (len(adata), 1))
    
    if do_sparse:
        adata.X = spr.csr_matrix(adata.X)
        
    if do_return:
        return adata


def plot_adata_cluster_properties(dict_cats_clusters={}, list_datasets=[], what='presence', cluster_name='cluster', axis_name='axis', list_clusters=None):
    if list_clusters is None:
        df_cols = [i for i in list(dict_cats_clusters.keys()) if len(i) == 2]
    else:
        df_cols = list_clusters
    
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


def make_gene_scoring(list_datasets=[], calculate_DEGs = True, group_name = 'cluster', value_ref = 'scores', select_method = 'pval', 
                      select_thres = 0.01, select_by='less', list_clusters=None):
    """In this function we are going to extract the selective markers for each population. 
       To do that we are going to apply a two-step procedure. 
       1) In the first step we are going to select the genes that are going to be evaluated. These
       are genes that, for each of the datasets, applies for the selection criterion. Then, the common set of genes between all the groups is created.
       The common set of genes is important to discriminate the importance of each dataset in the second step.
       2) Once the set of genes is created, we apply a weight for each of the datasets. We use a secondary reference value (score, logfold, pval...)
       which we assign for each gene for all datasets. To create a final score, we apply a ponderated mean of all values. We sum the values for all genes
       create a value per dataset. Then, we use these values to create a wieghted mean.
    """ 
        
    # Calculate DEGs if necessary
    for adata in list_datasets:        
        if calculate_DEGs:
            sc.tl.rank_genes_groups(adata, groupby=group_name, method='t-test_overestim_var')
    
    
    dict_scores = {}
    for cluster in list_clusters:
        # 1) Create the group of genes based on "value_ref".
        list_group_genes = []
        
        for adata in list_datasets:           
            list_terms = ['RPS', 'RPL', 'MT-', 'S100A', 'MYL', 'EIF', 'MALAT1']
            unsupported_genes = []
            for term in list_terms:
                unsupported_genes += [i for i in adata.var_names if i.startswith(term)]
                
            if cluster in adata.uns['rank_genes_groups']['names'].dtype.names:
                selected_genes = adata.uns['rank_genes_groups']['names'][cluster]

                if select_method == 'pval':
                    selected_vals = adata.uns['rank_genes_groups']['pvals_adj'][cluster]
                    selected_fold = adata.uns['rank_genes_groups']['logfoldchanges'][cluster]

                    mask = (selected_vals < select_thres) & (selected_fold > 0)
                else:
                    if select_by == 'less':
                        mask = (adata.uns['rank_genes_groups'][select_method][cluster] < select_thres)
                    else:
                        mask = (adata.uns['rank_genes_groups'][select_method][cluster] > select_thres)  

                list_group_genes += list([i for i in selected_genes[mask] if i not in unsupported_genes])
            
        set_group_genes = sorted(set(list_group_genes))
        
        # 2) Scoring genes
        list_columns = []
        for adata in list_datasets: 
            list_columns.append(f"{adata.obs['Author'].values[0]}-{int(adata.obs['Year'].values[0])}-{'/'.join(adata.obs['Condition'].cat.categories)}")
        df_cluster_score = pd.DataFrame(0, index=set_group_genes, columns=list_columns)
        
        for adata, adata_naming in zip(list_datasets, list_columns):
            if cluster in adata.uns['rank_genes_groups']['names'].dtype.names:
                genes = adata.uns['rank_genes_groups']['names'][cluster]
                if value_ref in ['scores', 'pvals_adj', 'logfoldchanges']:
                    values = adata.uns['rank_genes_groups'][value_ref][cluster]
                elif value_ref.endswith('expression'):
                    values = adata[:, genes].X.sum(0).A1 / len(adata)
                elif value_ref == 'expression_group':
                    values = np.dot(adata.obsp['distances'], adata[:, genes].X).sum(0).A1 / len(adata)
                    
                    
                values[values < 0] = 0  # this is to avoid getting strange stuff
                
                df_genes_values = pd.Series(values, index=genes)
                intersect_genes = np.intersect1d(genes, set_group_genes)

                df_cluster_score.loc[intersect_genes, adata_naming] = df_genes_values[intersect_genes]
        
        # the sqrt is to avoid putting too much weight to some odd dataset
        vec_score = df_cluster_score.values
        df_sum = df_cluster_score.sum().values
        vec_weights = 2 * (1/ (1 + np.exp(- df_sum / np.mean(df_sum) )) - 0.5)
        
        
        all_values_mean = np.dot(vec_score, vec_weights) / np.sum(vec_weights)        
        all_values_std = np.sqrt(np.dot((vec_score - all_values_mean.reshape(len(all_values_mean), 1)) ** 2, vec_weights) / np.sum(vec_weights))
        
        df_cluster_score['mean'], df_cluster_score['dev'] = all_values_mean, all_values_std ** (1/3)
        
        df_cluster_score['CV'] = df_cluster_score['mean'] / df_cluster_score['dev'] 
        
    
        dict_scores[cluster] = df_cluster_score
    
    return dict_scores


def make_gene_scoring_with_expr(value_ref = 'scores', expr_type='expression_group',  **kwargs):
    # We see that, when adding info about the gene expression, we can select genes that are underexpressed, but
    # localized in small populations, instead of genes expressed throughout the scene, although higher in the population of interest.
    dict_make_gene_scoring = make_gene_scoring(value_ref=value_ref, **kwargs)
    dict_make_gene_expression = make_gene_scoring(value_ref=expr_type,  **kwargs)
    
    dict_return = {}
    
    for cluster in dict_make_gene_scoring.keys():
        df_score = dict_make_gene_scoring[cluster].sort_values(by='mean', ascending=False)
        df_expr = dict_make_gene_expression[cluster].sort_values(by='mean', ascending=False)

        df_score['expr'] = df_expr['mean']
        df_score['expr_pow'] = df_score['expr'] ** 0.25  # This is a dampening factor, so that over or underexpressed genes do not disturb the ranking
        df_score['Z'] = df_score['mean'] / df_score['expr_pow']

        df_score = df_score[df_score['expr'] >= 0.035]  # Avoid genes with really really small expression
        
        dict_return[cluster] = df_score.sort_values(by='Z', ascending=False)
    
    return dict_return


def plot_score_graph(adatax, cluster_column='cluster'):
    df_cats_own = pd.DataFrame(index=adatax.obs_names, columns=['clusters', 'score'])
    for cluster in adatax.obs[cluster_column].cat.categories:
        adata_sub = adatax[adatax.obs[cluster_column] == cluster]
        try:
            df_cats_own.loc[adata_sub.obs_names, 'score'] = adata_sub.obs[f'{cluster_column}_{cluster}']
            df_cats_own.loc[adata_sub.obs_names, 'clusters'] = cluster
        except:
            pass

    df_cats_own = df_cats_own.sort_values('clusters')
    sns.barplot(x='clusters', y='score', data=df_cats_own, palette=adatax.uns[f'{cluster_column}_colors'])
    

def plot_UMAPS_gene(gene, list_datasets, list_names, n_cols=5, n_rows=None, cmap=magma):
    if n_rows is None:
        n_rows = int(math.ceil(len(list_datasets) / n_cols))
        
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    for ax in axs.ravel():
        ax.set_axis_off()

    for adata, name, idx in zip(list_datasets, list_names, range(len(list_datasets))):
        try:
            sc.pl.umap(adata, color=[gene], legend_loc='on data', show=False, ax = axs.ravel()[idx], 
                       title=name, size=15, cmap=cmap, frameon=False, use_raw=False)
        except:
            ...
            

def make_dicts_fraction_mean(genes, list_names, list_all_datasets, list_accepted_clusters, clusterby='cluster_robust'):
    dict_fraction_cells = {gene: pd.DataFrame(np.nan, index=list_names, columns=list_accepted_clusters) for gene in genes}
    dict_mean_exp = {gene: pd.DataFrame(np.nan, index=list_names, columns=list_accepted_clusters) for gene in genes}

    for adata, name in zip(list_all_datasets, list_names):
        genes_sub = [i for i in genes if i in adata.var_names]
        for cluster in set(adata.obs[clusterby]):
            counts = adata[adata.obs[clusterby] == cluster][:, genes_sub].X.toarray().copy()
            counts_frac = (counts > 0).sum(0) / counts.shape[0]
            counts[counts == 0] = np.nan
            counts_mean_exp = np.nanmean(counts, 0)

            for idx, gene in enumerate(genes_sub):
                dict_fraction_cells[gene].loc[name, cluster] = counts_frac[idx]
                dict_mean_exp[gene].loc[name, cluster] = counts_mean_exp[idx]

    for gene in genes:
        dict_fraction_cells_mean, dict_fraction_cells_std =  dict_fraction_cells[gene].mean(),  dict_fraction_cells[gene].std()
        dict_mean_exp_mean, dict_mean_exp_std =  dict_mean_exp[gene].mean(),  dict_mean_exp[gene].std()

        dict_fraction_cells[gene].loc['Mean'] = dict_fraction_cells_mean
        dict_fraction_cells[gene].loc['Std'] = dict_fraction_cells_std
        dict_mean_exp[gene].loc['Mean'] = dict_mean_exp_mean
        dict_mean_exp[gene].loc['Std'] = dict_mean_exp_std

        dict_fraction_cells[gene] = dict_fraction_cells[gene][list_accepted_clusters]
        dict_mean_exp[gene] = dict_mean_exp[gene][list_accepted_clusters]
    
    return dict_fraction_cells, dict_mean_exp


def plot_dotplot_gene(gene, dict_fraction_cells, dict_mean_exp, rotate=False):
    dfplot_frac = dict_fraction_cells[gene] ** 0.66
    dfplot_exp = dict_mean_exp[gene] 
    exp_norm_vals = (dfplot_exp.loc['Mean'] - min(dfplot_exp.loc['Mean'])) / (max(dfplot_exp.loc['Mean']) - min(dfplot_exp.loc['Mean']))
    fig, ax = plt.subplots(1, 1, figsize=(10, 1))
    ax.set_xticks(range(len(dfplot_frac.columns)))
    
    if rotate:
        ax.set_xticklabels(dfplot_frac.columns, rotation=40, ha='right')
    else:
        ax.set_xticklabels(dfplot_frac.columns)
    
    ax.set_yticks([0])
    ax.set_yticklabels([gene])
    ax.set_ylim([-0.1, 0.1])
    plt.scatter(range(len(dfplot_frac.columns)), [0] * len(dfplot_frac.columns), s=dfplot_frac.loc['Mean'] * 550, c=[cm.OrRd(i) for i in exp_norm_vals], 
                linewidths=0.5, edgecolor='#878787', alpha = [max(0, i) for i in 1 - dict_fraction_cells[gene].loc['Std'] ** 0.75])
    
    plt.plot([-0.3, len(dfplot_frac.columns) - 0.3], [0, 0], c="#676767", linewidth=0.7, alpha=0.3)
    plt.plot([-0.3, len(dfplot_frac.columns) - 0.3], [0.025, 0.025], c="#676767", linewidth=0.7, alpha=0.3)
    plt.plot([-0.3, len(dfplot_frac.columns) - 0.3], [-0.025, -0.025], c="#676767", linewidth=0.7, alpha=0.3)
    


    
    
def plot_dotplot_list_genes(list_genes, dict_fraction_cells, dict_mean_exp, rotate=False, figsize=(10, 20), do_return=False):
    dfplot_frac = dict_fraction_cells[list_genes[0]] ** 0.66
    
    fig, ax = plt.subplots(1, 1,  figsize=figsize)
    
    ax.set_xticks(range(len(dfplot_frac.columns)))

    if rotate:
        ax.set_xticklabels(dfplot_frac.columns, rotation=40, ha='right')
    else:
        ax.set_xticklabels(dfplot_frac.columns)

    ax.set_yticks(range(len(list_genes)))
    ax.set_yticklabels(list_genes)
    ax.set_ylim([-1.1, len(list_genes) + 0.1])
    
    for gene_idx, gene in enumerate(list_genes):
        dfplot_frac = dict_fraction_cells[gene] ** 0.66
        dfplot_exp = dict_mean_exp[gene] 
        exp_norm_vals = (dfplot_exp.loc['Mean'] - min(dfplot_exp.loc['Mean'])) / (max(dfplot_exp.loc['Mean']) - min(dfplot_exp.loc['Mean']))
        
        ax.scatter(range(len(dfplot_frac.columns)), [gene_idx] * len(dfplot_frac.columns), s=dfplot_frac.loc['Mean'] * 550, 
                   c=[cm.OrRd(i) for i in exp_norm_vals], linewidths=0.5, edgecolor='#878787', 
                   alpha = [max(0, i) for i in 1 - dict_fraction_cells[gene].loc['Std'] ** 0.75])
    
    plt.gca().invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    if do_return:
        return ax