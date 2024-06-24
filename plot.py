import warnings
import itertools

import pandas as pd
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import colors
import seaborn as sns
from statannotations.Annotator import Annotator
from scipy.stats import pearsonr, false_discovery_control
import scipy.cluster.hierarchy as sch

def plot_stb_cv(adata, x:str = 'pool_cv', y:str = 'pool_stability_cv', z:str='pool_mean', hue:str = 'uclustering_cv_stb_labels', palette:str = None, legend:bool = False, figsize:tuple = (15,10), median_line:bool = True):
    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(3,1)
    # gs_top = GridSpec(2, 3, width_ratios=[8, 1, 8], height_ratios=[1, 5], wspace=.02, hspace=.02)
    gs_top = gs[0].subgridspec(2, 3, width_ratios=[8, 1, 8], height_ratios=[1, 5], wspace=.02, hspace=.02)
    ax_clu = fig.add_subplot(gs_top[1,0])
    ax_y = fig.add_subplot(gs_top[1,1])
    ax_x = fig.add_subplot(gs_top[0,0])
    ax_y_bar = fig.add_subplot(gs_top[0,2])
    ax_y_violin = fig.add_subplot(gs_top[1,2])

    gs_middle = gs[1].subgridspec(2, 3, width_ratios=[8, 1, 8], height_ratios=[1, 5], wspace=.02, hspace=.02)
    ax_clu2 = fig.add_subplot(gs_middle[1,0])
    ax_y2 = fig.add_subplot(gs_middle[1,1])
    ax_x2 = fig.add_subplot(gs_middle[0,0])
    ax_y_bar2 = fig.add_subplot(gs_middle[0,2])
    ax_y_violin2 = fig.add_subplot(gs_middle[1,2])

    gs_bottom = gs[2].subgridspec(2, 3, width_ratios=[8, 1, 8], height_ratios=[1, 5], wspace=.02, hspace=.02)
    ax_clu3 = fig.add_subplot(gs_bottom[1,0])
    ax_y3 = fig.add_subplot(gs_bottom[1,1])
    ax_x3 = fig.add_subplot(gs_bottom[0,0])
    ax_y_bar3 = fig.add_subplot(gs_bottom[0,2])
    ax_y_violin3 = fig.add_subplot(gs_bottom[1,2])

    hue_order = adata.var[hue].unique()
    if hue == None:
        palette = None
        cmap = None
    else:
        cmap = plt.get_cmap(palette, len(hue_order))
        cmap = [colors.to_hex(cmap(i)) for i in range(len(hue_order))]
        cmap = dict(zip(hue_order,cmap))

    # TOP grid
    sns.scatterplot(x=x, y=y, hue=hue, palette=cmap, data=adata.var, ax=ax_clu)
    ax_clu.get_legend().set_visible(legend)
    ax_clu.spines[['right', 'top']].set_visible(False)

    sns.histplot(x=x, ec='white', data=adata.var, kde=True, ax=ax_x)
    ax_x.spines[['left','right', 'top']].set_visible(False)
    ax_x.set_xlabel(None)
    ax_x.set_ylabel(None)
    ax_x.set_xticklabels([])
    ax_x.set_yticklabels([])
    ax_x.tick_params(left = False)
    ax_x.grid(False)

    sns.histplot(y=y, ec='white', data=adata.var, kde=True, ax=ax_y)
    ax_y.spines[['right', 'top','bottom']].set_visible(False)
    ax_y.set_ylabel(None)
    ax_y.set_xlabel(None)
    ax_y.set_yticklabels([])
    ax_y.set_xticklabels([])
    ax_y.tick_params(bottom = False)
    ax_y.grid(False)

    order = adata.var.groupby(hue).median(numeric_only=True).sort_values(y).index
    sns.countplot(x=hue, order=order, color='gray', data=adata.var, ax=ax_y_bar)
    ax_y_bar.set_xticklabels([])
    ax_y_bar.set_xlabel(None)
    ax_y_bar.spines[['left','right', 'top']].set_visible(False)
    ax_y_bar.set_ylabel('Qty.')

    sns.stripplot(x=hue, y=y, order=order, color='black', s=1, data=adata.var, ax=ax_y_violin)
    sns.boxplot(x=hue, y=y, order=order, palette=cmap, data=adata.var, ax=ax_y_violin)
    ax_y_violin.set_ylabel(None)
    ax_y_violin.set_yticklabels([])
    aux_md = adata.var.groupby(hue).median(numeric_only=True)
    arg_max = len(aux_md.loc[aux_md[y]<=adata.var[y].median(numeric_only=True)][y]) - .5
    ax_y_violin.axvline(arg_max, ls=':', lw=1,color='gray')

    # Middle grid
    sns.scatterplot(x=x, y=z, hue=hue, palette=cmap, data=adata.var, ax=ax_clu2)
    ax_clu2.get_legend().set_visible(legend)
    ax_clu2.spines[['right', 'top']].set_visible(False)

    sns.histplot(x=x, ec='white', data=adata.var, kde=True, ax=ax_x2)
    ax_x2.spines[['left','right', 'top']].set_visible(False)
    ax_x2.set_xlabel(None)
    ax_x2.set_ylabel(None)
    ax_x2.set_xticklabels([])
    ax_x2.set_yticklabels([])
    ax_x2.tick_params(left = False)
    ax_x2.grid(False)

    sns.histplot(y=z, ec='white', data=adata.var, kde=True, ax=ax_y2)
    ax_y2.spines[['right', 'top','bottom']].set_visible(False)
    ax_y2.set_ylabel(None)
    ax_y2.set_xlabel(None)
    ax_y2.set_yticklabels([])
    ax_y2.set_xticklabels([])
    ax_y2.tick_params(bottom = False)
    ax_y2.grid(False)

    order = adata.var.groupby(hue).median(numeric_only=True).sort_values(z).index
    sns.countplot(x=hue, order=order, color='gray', data=adata.var, ax=ax_y_bar2)
    ax_y_bar2.set_xticklabels([])
    ax_y_bar2.set_xlabel(None)
    ax_y_bar2.spines[['left','right', 'top']].set_visible(False)
    ax_y_bar2.set_ylabel('Qty.')

    sns.stripplot(x=hue, y=z, order=order, color='black', s=1, data=adata.var, ax=ax_y_violin2)
    sns.boxplot(x=hue, y=z, order=order, palette=cmap, data=adata.var, ax=ax_y_violin2)
    ax_y_violin2.set_ylabel(None)
    ax_y_violin2.set_yticklabels([])
    aux_md = adata.var.groupby(hue).median(numeric_only=True)
    arg_max = len(aux_md.loc[aux_md[z]<=adata.var[z].median(numeric_only=True)][z]) - .5
    ax_y_violin2.axvline(arg_max, ls=':', lw=1,color='gray')

    # Bottom grid
    sns.scatterplot(x=z, y=x, hue=hue, palette=cmap, data=adata.var, ax=ax_clu3)
    ax_clu3.get_legend().set_visible(legend)
    ax_clu3.spines[['right', 'top']].set_visible(False)

    sns.histplot(x=z, ec='white', data=adata.var, kde=True, ax=ax_x3)
    ax_x3.spines[['left','right', 'top']].set_visible(False)
    ax_x3.set_xlabel(None)
    ax_x3.set_ylabel(None)
    ax_x3.set_xticklabels([])
    ax_x3.set_yticklabels([])
    ax_x3.tick_params(left = False)
    ax_x3.grid(False)

    sns.histplot(y=x, ec='white', data=adata.var, kde=True, ax=ax_y3)
    ax_y3.spines[['right', 'top','bottom']].set_visible(False)
    ax_y3.set_ylabel(None)
    ax_y3.set_xlabel(None)
    ax_y3.set_yticklabels([])
    ax_y3.set_xticklabels([])
    ax_y3.tick_params(bottom = False)
    ax_y3.grid(False)

    order = adata.var.groupby(hue).median(numeric_only=True).sort_values(x).index
    sns.countplot(x=hue, order=order, color='gray', data=adata.var, ax=ax_y_bar3)
    ax_y_bar3.set_xticklabels([])
    ax_y_bar3.set_xlabel(None)
    ax_y_bar3.spines[['left','right', 'top']].set_visible(False)
    ax_y_bar3.set_ylabel('Qty.')

    sns.stripplot(x=hue, y=x, order=order, color='black', s=1, data=adata.var, ax=ax_y_violin3)
    sns.boxplot(x=hue, y=x, order=order, palette=cmap, data=adata.var, ax=ax_y_violin3)
    ax_y_violin3.set_ylabel(None)
    ax_y_violin3.set_yticklabels([])
    aux_md = adata.var.groupby(hue).median(numeric_only=True)
    arg_max = len(aux_md.loc[aux_md[x]<=adata.var[x].median(numeric_only=True)][x]) - .5
    ax_y_violin3.axvline(arg_max, ls=':', lw=1,color='gray')

    if median_line:
        ax_clu.axvline(np.median(adata.var[x]), ls='--', lw=1,color='black')
        ax_clu.axhline(np.median(adata.var[y]), ls='--', lw=1, color='black')
        ax_y_violin.axhline(np.median(adata.var[y]), ls='--', lw=1, color='black')

        ax_clu2.axvline(np.median(adata.var[x]), ls='--', lw=1,color='black')
        ax_clu2.axhline(np.median(adata.var[z]), ls='--', lw=1, color='black')
        ax_y_violin2.axhline(np.median(adata.var[z]), ls='--', lw=1, color='black')

        ax_clu3.axvline(np.median(adata.var[z]), ls='--', lw=1,color='black')
        ax_clu3.axhline(np.median(adata.var[x]), ls='--', lw=1, color='black')
        ax_y_violin3.axhline(np.median(adata.var[x]), ls='--', lw=1, color='black')

    return None

def plot_corr(adata, layer:str = None, col_labels:str = 'uclustering_cv_stb_labels', labels:list = None, r_pearson_lim:float = 0.5, p_value_lim:float = 0.05, fdr:bool = True, figsize:tuple = (7,7)):
    if labels == None:
        labels = []
        labels.append(adata.var.groupby(col_labels).median(numeric_only=True).sort_values('pool_cv').index[0])

    if layer != None:
        aux = adata[:,adata.var.[col_labels].isin(labels)].layers[layer]
    else:
        aux = adata[:,adata.var.[col_labels].isin(labels)].X

    cols_ = list(itertools.combinations_with_replacement(aux.columns,2, ))
    P_list = []
    pv_list = []
    pair = []
    for col in cols_:
        pair.append(','.join(col))
        P_list.append(pearsonr(aux[col[0]], aux[col[1]]).statistic)
        pv_list.append(pearsonr(aux[col[0]], aux[col[1]]).pvalue)

    if fdr:
        pv_list = false_discovery_control(pv_list)
    aux = pd.DataFrame([pair, P_list,pv_list], index=['gene_pair','R','pvalue']).T
    aux[['Gene_name_x', 'Gene_name_y']] = aux.gene_pair.str.split(',', expand=True)
    aux_2 = aux.copy()
    aux_2.columns = ['gene_pair', 'R', 'pvalue', 'Gene_name_y', 'Gene_name_x']#
    aux = pd.concat([aux, aux_2])
    aux_R = aux.drop_duplicates().pivot(columns='Gene_name_x', index='Gene_name_y', values='R')

    Z=sch.linkage(aux_R.astype(float), method='ward')
    order = sch.leaves_list(Z)
    n_gene_dict = dict(zip(aux_R.columns[order],range(len(aux_R.columns))))

    aux['Gene_name_x'] = aux.Gene_name_x.map(n_gene_dict)
    aux['Gene_name_y'] = aux.Gene_name_y.map(n_gene_dict)
    aux['R'] = np.where((aux.R.abs().values>=r_pearson_lim), 'R Pearson ≥'+str(r_pearson_lim), 'R Pearson <'+str(r_pearson_lim))
    aux['pvalue'] = np.where((aux.pvalue.abs().values<=p_value_lim), 'padj ≤'+str(p_value_lim), 'padj >'+str(p_value_lim))


    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    sns.scatterplot(x='Gene_name_x', y='Gene_name_y', hue='R', marker='o', size='pvalue', palette=['green', 'black'], edgecolor='grey', data=aux, ax=ax)
    ax.set_xticks(range(len(n_gene_dict.keys())), aux_R.columns[order], rotation=90, style='italic')
    ax.set_yticks(range(len(n_gene_dict.keys())), aux_R.columns[order], style='italic')
    ax.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.legend(bbox_to_anchor=(.85, -.175), ncol=2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=np.array(handles)[[1,2,4,5]].tolist(), labels=np.array(labels)[[1,2,4,5]].tolist(), bbox_to_anchor=(1., 1.), ncol=2)
    plt.grid(True, alpha=.25)
    plt.gca().set_axisbelow(True)

    return None


# import warnings
#
# import pandas as pd
# import numpy as np
# import anndata as ad
#
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from matplotlib import cm
# import seaborn as sns
#
# def plot_stb_cv(adata, x:str = 'pool_cv', y:str = 'pool_stability_cv', hue:str = 'pool_mean', palette:str = None, legend:bool = False, figsize:tuple = (10,5), median_line:bool = True):
#     fig = plt.figure(figsize=figsize)
#     gs = GridSpec(3, 3, width_ratios=[8, 1, 9], height_ratios=[1, 5, 5], wspace=.02, hspace=.02)
#
#     ax_clu = fig.add_subplot(gs[1,0])
#     ax_hist_cv = fig.add_subplot(gs[1,1])
#     ax_hist_mean = fig.add_subplot(gs[0,0])
#
#     if hue == None:
#         palette = None
#     sns.scatterplot(x=x, y=y, hue=hue, palette=palette, data=adata.var, ax=ax_clu)
#     ax_clu.get_legend().set_visible(legend)
#     ax_clu.spines[['right', 'top']].set_visible(False)
#
#     sns.histplot(x=x, ec='white', data=adata.var, kde=True, ax=ax_hist_mean)
#     ax_hist_mean.spines[['left','right', 'top']].set_visible(False)
#     ax_hist_mean.set_xlabel(None)
#     ax_hist_mean.set_ylabel(None)
#     ax_hist_mean.set_xticklabels([])
#     ax_hist_mean.set_yticklabels([])
#     ax_hist_mean.tick_params(left = False)
#     ax_hist_mean.grid(False)
#
#     sns.histplot(y=y, ec='white', data=adata.var, kde=True, ax=ax_hist_cv)
#     ax_hist_cv.spines[['right', 'top','bottom']].set_visible(False)
#     ax_hist_cv.set_ylabel(None)
#     ax_hist_cv.set_xlabel(None)
#     ax_hist_cv.set_yticklabels([])
#     ax_hist_cv.set_xticklabels([])
#     ax_hist_cv.tick_params(bottom = False)
#     ax_hist_cv.grid(False)
#
#     if median_line:
#         ax_clu.axvline(np.median(adata.var[x]), ls='--', lw=1,color='black')
#         ax_clu.axhline(np.median(adata.var[y]), ls='--', lw=1, color='black')
#
#     return None
