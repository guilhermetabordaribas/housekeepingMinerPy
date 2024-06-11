import itertools
import numpy as np
import pandas as pd
import anndata as ad
import pygad
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sknetwork.clustering import Louvain

def exprs_cv(adata, layer:str = None, groups_col:str = None, return_mean_per_group:bool = False, return_std_per_group:bool = False, return_cv_per_group:bool = False):
    """
    Calculate the coefficient (cv) of variation of expression for each gene. To give the same weight for different groups, you can infrom groups_col.
    So a pooled cv will be calculated considering the same weight for each group.

    Parameters
    ----------
    adata : Anndata
    layer : string, optional
        layer of AnnData object to be used to transform.
        If layer is not informed, adata.X will be used.
    groups_col: string
        groups_col to perform the stratified calculation of cv. It must be a column at adata.obs annotations.
        If None, the calculation will not give the same weight for each group. The group with more samples will have greater weight. The name of column will be simple_cv instead pooled_cv.
    return_mean_per_group: bool
        If True, columns of mean calculation for each group will be stored in a adata.var column.
    return_std_per_group: bool
        If True, columns of standard deviation (std) calculation for each group will be stored in a adata.var column.
    return_std_per_group: bool
        If True, columns of cv calculation for each group will be stored in a adata.var column.

    Returns
    -------
    adata
        Return the adata with additional columns ['pool_mean', 'pool_std', 'pool_cv'].

    """
    if groups_col == None:
        if layer != None:
            X_ = adata.layers[layer]
        else:
            X_ = adata.X

        adata.var['simple_mean'] = np.mean(X_, axis=0)
        adata.var['simple_std'] = np.std(X_, axis=0)
        adata.var['simple_cv'] = adata.var['std'].values / adata.var['mean'].values
        warnings.warn('Since groups_col == None, CV is calculated considering all samples without stratification per group.')

        return adata

    else:
        cols_aux = []
        for gc in adata.obs[groups_col].unique():
            if layer != None:
                X_ = adata[adata.obs[groups_col]==gc,:].layers[layer]
            else:
                X_ = adata[adata.obs[groups_col]==gc,:].X

            adata.var['mean_'+str(gc)] = np.mean(X_, axis=0)
            adata.var['std_'+str(gc)] = np.std(X_, axis=0)
            if return_cv_per_group:
                adata.var['cv_'+str(gc)] = adata.var['std_'+str(gc)].values / adata.var['mean_'+str(gc)].values

        adata.var['pool_mean'] = adata.var[['mean_'+str(c) for c in adata.obs[groups_col].unique()]].mean(axis=1)
        adata.var['pool_std'] = np.sqrt( np.square(adata.var[['std_'+str(c) for c in adata.obs[groups_col].unique()]]).mean(axis=1) )
        adata.var['pool_cv'] = adata.var['pool_std'].values / adata.var['pool_mean'].values

        if not return_mean_per_group:
            adata.var = adata.var.drop(['mean_'+str(c) for c in adata.obs[groups_col].unique()], axis=1)
        if not return_std_per_group:
            adata.var = adata.var.drop(['std_'+str(c) for c in adata.obs[groups_col].unique()], axis=1)

        return adata

def stability_cv(adata, layer:str = None, groups_col:str = None, return_stb_cv_per_group:bool = False):
    """
    Calculate the average coefficient (cv) of variation of stability for each pair of genes. To give the same weight for different groups, you can infrom groups_col.
    So a pooled stability cv will be calculated considering the same weight for each group.

    Parameters
    ----------
    adata : Anndata
    layer : string, optional
        layer of AnnData object to be used to transform.
        If layer is not informed, adata.X will be used.
    groups_col: string
        groups_col to perform the stratified calculation of cv. It must be a column at adata.obs annotations.
        If None, the calculation will not give the same weight for each group. The group with more samples will have greater weight. The name of column will be simple_cv instead pooled_cv.

    Returns
    -------
    adata
        Return the adata with additional column ['pool_stability_cv'].

    """
    # Since the stability ratio is calculate by sample, it is independent of groups_col.
    # Only after calculation of mean, std and cv requires stratification process.
    if layer != None:
        X_ = adata.layers[layer]
    else:
        X_ = adata.X

    cols_ = []
    df_mean_ = []
    df_std_ = []
    if groups_col == None:
        pw_dist = pairwise_distances(X_[:,:].T)
        adata.var['simple_stability_cv'] = pw_dist.std(axis=0) / pw_dist.mean(axis=0)
        # mean_X_ = []
        # std_X_ = []
        # for i in range(X_.shape[1]):
        #     mean_X_.extend(np.mean(X_[:,i:] - X_[:,i,None], axis=1)[0])
        #     std_X_.extend(np.std(X_[:,i:] - X_[:,i,None], axis=1)[0])
        #     cols_.extend([','.join(p) for p in itertools.product([adata.var.index[i]],adata.var[i:].index)])
        # df_mean_.append(mean_X_)
        # df_std_.append(std_X_)
    else:
        j = 0
        groups = adata.obs[groups_col].unique()
        print('Computing groups data:', end=' ')
        for i,gc in enumerate(groups):
            print(i, end=' ')
            idx = np.where(adata.obs[groups_col]==gc)
            pw_dist = pairwise_distances(X_[idx,:][0].T)
            df_mean_.append( pw_dist.mean(axis=0) )
            df_std_.append( pw_dist.std(axis=0) )

        adata.var['pool_stability_cv'] = (np.array(df_std_) / np.array(df_mean_)).mean(axis=0)

        if return_stb_cv_per_group:
            aux = pd.DataFrame(np.array(df_std_) / np.array(df_mean_)).T
            aux.columns = ['gc_'+str(g) for g in groups]
            adata.var[aux.columns] = aux
            # mean_X_ = []
            # std_X_ = []
            # for i in range(X_.shape[1]):
            #     mean_X_.extend(np.mean(X_[idx,i:] - X_[idx,i,None], axis=1)[0])
            #     std_X_.extend(np.std(X_[idx,i:] - X_[idx,i,None], axis=1)[0])
            #     if j == 0:
            #         cols_.extend([','.join(p) for p in itertools.product([adata.var.index[i]],adata.var[i:].index)])
            # j+=1
            # df_mean_.append(mean_X_)
            # df_std_.append(std_X_)

    # aux = pd.DataFrame(df_std_, columns=cols_, index=groups) / pd.DataFrame(df_mean_, columns=cols_, index=groups)
    # del(df_std_)
    # del(df_mean_)
    # aux = aux[[c for c in aux.columns if (len(set(c.split(',')))>1)]].abs()
    # aux = aux.mean(axis=0).reset_index()
    # aux.columns = ['gene', 'stability_cv']
    # aux[['G1','G2']] = aux['gene'].str.split(',', expand=True)
    # aux = pd.concat([aux[['G1','stability_cv']].set_index('G1'), aux[['G2','stability_cv']].set_index('G2')], axis=0).reset_index()
    # aux.columns = ['gene', 'stability_cv']
    # adata.var['pool_stability_cv'] = aux.groupby('gene').mean().loc[adata.var.index,:]

    return adata

def uclustering_cv_stb(adata, cv_col:str = None, stb_col:str = None, scaler_object = None, nearestNeighbors_object = None, louvain_object = None):

    if nearestNeighbors_object == None:
        nearestNeighbors_object = NearestNeighbors()

    if louvain_object == None:
        louvain_object = Louvain(random_state=42)

    if cv_col == None:
        if ('pool_cv' in adata.var.columns):
            cv_col = 'pool_cv'
        if ('simple_cv' in adata.var.columns):
            cv_col = 'simple_cv'

    if stb_col == None:
        if ('pool_stability_cv' in adata.var.columns):
            stb_col = 'pool_stability_cv'
        elif ('simple_stability_cv' in adata.var.columns):
            stb_col = 'simple_stability_cv'

    cl_X = adata.var[[cv_col, stb_col]]

    if scaler_object != None:
        cl_X = scaler_object.fit_transform(cl_X)

    nearestNeighbors_object.fit(cl_X)
    labels = louvain_object.fit_predict(nearestNeighbors_object.kneighbors_graph())

    adata.var['uclustering_cv_stb_labels'] = labels

    return adata

def sclustering_cv_stb(adata, cv_col:str = None, stb_col:str = None, scaler_object = None, kMeans = None):

    if kMeans == None:
        kMeans = kMeans()

    if cv_col == None:
        if ('pool_cv' in adata.var.columns):
            cv_col = 'pool_cv'
    if stb_col == None:
        if ('pool_stability_cv' in adata.var.columns):
            stb_col = 'pool_stability_cv'

    cl_X = adata.var[[cv_col, stb_col]]

    if scaler_object != None:
        cl_X = scaler_object.fit_transform(cl_X)

    kMeans.fit(cl_X)
    labels = kmeans.labels_

    adata.var['sclustering_cv_stb_labels'] = labels

    return adata


def hkg_selection_ga(adata, layer:str = None, outlier_threshold:float = .9, fitness_function:str = None):
    if layer != None:
        X_ = adata.layers[layer]
    else:
        X_ = adata.X

    if fitness_function == 'minimize_groups_qty':
        def fitness_func(ga_instance, solution, solution_idx):
            clusterer = hdbscan.HDBSCAN().fit(X_[:, np.where( solution )[0]] )
            labels = clusterer.labels_
            fitness = 1 / np.unique(labels).shape[0]
            return fitness
    elif fitness_function == 'minimize_outliers':
        def fitness_func(ga_instance, solution, solution_idx):
            clusterer = hdbscan.HDBSCAN().fit(X_[:, np.where( solution )[0]] )
            threshold = np.quantile(clusterer.outlier_scores_, q=.9)
            inliers = np.where(clusterer.outlier_scores_ <= threshold)[0].shape[0]
            fitness = inliers
            return fitness

    num_generations = 100
    num_parents_mating = 2

    sol_per_pop = 8
    num_genes = len(bm)

    init_range_low = 0
    init_range_high = 2

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 10

    gene_type = int

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           # init_range_low=init_range_low,
                           # init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           gene_type=gene_type,
                           gene_space=[0, 1]
                          )

    ga_instance.run()
    # solution, solution_fitness, solution_idx = ga_instance.best_solution()
    return ga_instance.best_solution()

# def fitness_func(ga_instance, solution, solution_idx):
#     lb = LabelBinarizer()
#     # X = df_cnt.iloc[:,:-1].values[:,np.where(solution)[0]]
#     # y = lb.fit_transform(df_cnt.ar.values).flatten()
#     X_train, X_test, y_train, y_test = train_test_split(X_o[:,np.where(solution)[0]], y_o, test_size=0.33, random_state=42)
#
#     neigh = KNeighborsClassifier(n_neighbors=2)
#     neigh.fit(X_train, y_train)
#     y_pred = neigh.predict(X_test)
#     # fitness = 1 / ( f1_score(y_test, y_pred) * np.mean(np.var(X_o[:,np.where(solution)[0]], axis=0))) # * solution.sum()
#     fitness = 1 / ( f1_score(y_test, y_pred) * gmean(np.var(X_o[:,np.where(solution)[0]], axis=0)) )# * solution.sum()
#     # fitness = inliers
#     return fitness
#
# fitness_function = fitness_func
#
# num_generations = 100
# num_parents_mating = 2
#
# sol_per_pop = 8
# if intial:
#     num_genes = len(bm)
# else:
#     num_genes = 1176
#
# init_range_low = 0
# init_range_high = 2
#
# parent_selection_type = "sss"
# keep_parents = 1
#
# crossover_type = "single_point"
#
# mutation_type = "random"
# mutation_percent_genes = 10
#
# gene_type=int
#
# ga_instance = pygad.GA(num_generations=num_generations,
#                        num_parents_mating=num_parents_mating,
#                        fitness_func=fitness_function,
#                        sol_per_pop=sol_per_pop,
#                        num_genes=num_genes,
#                        # init_range_low=init_range_low,
#                        # init_range_high=init_range_high,
#                        parent_selection_type=parent_selection_type,
#                        keep_parents=keep_parents,
#                        crossover_type=crossover_type,
#                        mutation_type=mutation_type,
#                        mutation_percent_genes=mutation_percent_genes,
#                        gene_type=gene_type,
#                        gene_space=[0, 1]
#                       )
#
# ga_instance.run()
# # print(ga_instance.initial_population[0])
# # print(ga_instance.population)
#
# solution, solution_fitness, solution_idx = ga_instance.best_solution()
# print("Parameters of the best solution : {solution}".format(solution=solution))
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
# print(np.where( solution )[0].shape)
# prediction = numpy.sum(numpy.array(function_inputs)*solution)
# print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
