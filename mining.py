import warnings
import itertools

import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sknetwork.clustering import Louvain
import hdbscan
import pygad
from boruta import BorutaPy

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

def gene_gini_coeff(adata, layer:str = None):
    """
    G = 1 + 1/n - 2*sum_i(rank_k*x_i) / n*sum_i(x_i)
    """

    if layer != None:
        X_ = adata.layers[layer]
    else:
        X_ = adata.X

    gini_list = []
    for i in range(X_.shape[1]):
        x = np.abs(X_[:,i])
        n = len(x)
        s = x.sum()
        rank = np.argsort(np.argsort(-x))
        gini_list.append( 1 - (2.0 * (rank*x).sum() + s)/(n*s) )

    adata.var['gini_coefficient'] = gini_list
    return adata

def uclustering_cv_stb(adata, cl_cols:list = [], scaler_object = None, nearestNeighbors_object = None, louvain_object = None, resolution:float = 1):

    if nearestNeighbors_object == None:
        nearestNeighbors_object = NearestNeighbors()

    if louvain_object == None:
        louvain_object = Louvain(random_state=42, resolution=resolution)

    # if cv_col == None:
    #     if ('pool_cv' in adata.var.columns):
    #         cv_col = 'pool_cv'
    #     if ('simple_cv' in adata.var.columns):
    #         cv_col = 'simple_cv'
    #
    # if stb_col == None:
    #     if ('pool_stability_cv' in adata.var.columns):
    #         stb_col = 'pool_stability_cv'
    #     elif ('simple_stability_cv' in adata.var.columns):
    #         stb_col = 'simple_stability_cv'
    if not cl_cols:
        if ('pool_cv' in adata.var.columns):
            cl_cols.append('pool_cv')
        elif ('simple_cv' in adata.var.columns):
            cl_cols.append('simple_cv')

        if ('pool_stability_cv' in adata.var.columns):
            cl_cols.append('pool_stability_cv')
        elif ('simple_stability_cv' in adata.var.columns):
            cl_cols.append('simple_stability_cv')

        if ('pool_mean' in adata.var.columns):
            cl_cols.append('pool_mean')
        elif ('simple_mean' in adata.var.columns):
            cl_cols.append('simple_mean')

    if not cl_cols:
        raise Warning("You inform a empty 'cl_col' argument, but no columns for cv, stability_cv or mean were found. Please, consider run 'stability_cv()' and 'exprs_cv()' funciotns")

    cl_X = adata.var[cl_cols]

    if scaler_object != None:
        cl_X = scaler_object.fit_transform(cl_X)

    nearestNeighbors_object.fit(cl_X)
    labels = louvain_object.fit_predict(nearestNeighbors_object.kneighbors_graph())

    adata.var['uclustering_cv_stb_labels'] = labels

    return adata

def sclustering_cv_stb(adata, cl_cols:list = [], scaler_object = None, kMeans = None):

    if kMeans == None:
        kMeans = kMeans()

    # if cv_col == None:
    #     if ('pool_cv' in adata.var.columns):
    #         cv_col = 'pool_cv'
    # if stb_col == None:
    #     if ('pool_stability_cv' in adata.var.columns):
    #         stb_col = 'pool_stability_cv'
    if not cl_cols:
        if ('pool_cv' in adata.var.columns):
            cl_cols.append('pool_cv')
        elif ('simple_cv' in adata.var.columns):
            cl_cols.append('simple_cv')

        if ('pool_stability_cv' in adata.var.columns):
            cl_cols.append('pool_stability_cv')
        elif ('simple_stability_cv' in adata.var.columns):
            cl_cols.append('simple_stability_cv')

        if ('pool_mean' in adata.var.columns):
            cl_cols.append('pool_mean')
        elif ('simple_mean' in adata.var.columns):
            cl_cols.append('simple_mean')

    if not cl_cols:
        raise Warning("You inform a empty 'cl_col' argument, but no columns for cv, stability_cv or mean were found. Please, consider run 'stability_cv()' and 'exprs_cv()' funciotns")


    cl_X = adata.var[cl_cols]

    if scaler_object != None:
        cl_X = scaler_object.fit_transform(cl_X)

    kMeans.fit(cl_X)
    labels = kmeans.labels_

    adata.var['sclustering_cv_stb_labels'] = labels

    return adata

def tost(adata, layer:str = None, conditions:str = None, vars_list:str = [], cohens_d:float = .5, is_parametric:bool = False, is_paired:bool = False):
    if layer != None:
        X_ = adata.to_df(layer)
    else:
        X_ = adata.to_df()
    # aux =
    dict_pairs = {p:[] for p in itertools.combinations(aux.conditions.unique(), 2)}


def hkg_selection_ga(adata, layer:str = None, outlier_threshold:float = .9, fitness_function:str = 'minimize_outliers', fitness_function_model = None, y:str = None):
    if layer != None:
        X_ = adata.layers[layer]
    else:
        X_ = adata.X

    lb = LabelBinarizer()
    y_ = lb.fit_transform(adata.obs[y].values)



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
    elif fitness_function == 'minimize_accuarcy':
        def fitness_func(ga_instance, solution, solution_idx):
            if fitness_function_model != None:
                clf = fitness_function_model
            else:
                clf = SVC()
                clf.fit(X_, y)
            score_ = clf.score(X_, y_)

            if  score_ == 0.5:
                score_ = 0.5 + 0.000001

            fitness = 1 / np.abs(0.5 - score_)
            return fitness

    num_generations = 100
    num_parents_mating = 2

    sol_per_pop = 8
    num_genes = X_.shape[1]#len(bm)

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
                           fitness_func=fitness_func,
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


def boruta_selection(adata, layer:str = None, class_col:str = None, scaler = None, rf_model = None, class_weight:list = None, random_state:int = 42):
    if layer != None:
        X_ = adata.layers[layer]
    else:
        X_ = adata.X

    if class_col == None:
        raise Exception("Boruta feature selection requires a class_col argument, with, at least, two different classes.")
    else:
        y_ = adata.obs[class_col].values

    if scaler != None:
        X_ = scaler.fit_transform(X_)

    if class_weight == None:
        unique_ = np.unique(y_)
        class_weight = dict(zip(unique_,compute_class_weight(class_weight="balanced", classes=unique_, y=y_)))

    if rf_model == None:
        rf_model = RandomForestClassifier(class_weight=class_weight)

    feat_selector = BorutaPy(rf_model, n_estimators='auto', verbose=0, random_state=random_state)
    feat_selector.fit(X_, y_)

    feature_ranks = list(zip(adata.var.index,
                         feat_selector.ranking_,
                         feat_selector.support_))

    result = {'genes':[], 'rank':[], 'support':[]}
    for feat in feature_ranks:
        if feat[2]:
            result['genes'].append(feat[0])
            result['rank'].append(feat[1])
            result['support'].append(feat[2])
            # print('Feature: {:<25} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))
    return result

def set_boruta_selection(adata, layer:str = None, class_col:str = None, scaler = None,  rf_model = None, random_state:int = 42, class_weight:list = None, n_set:int = 5, sample_size:int = None, replace:bool = False):
    if class_col == None:
        raise Exception("Boruta feature selection requires a class_col argument, with, at least, two different classes.")
    else:
        y_ = adata.obs[class_col].values

    results = []
    for i,set in enumerate(set_balance_resample(y_, n_set=n_set, random_state=random_state, sample_size=sample_size, replace=replace)):
        results.append(boruta_selection(adata, layer=layer, rf_model=rf_model, class_col=class_col, class_weight=class_weight, scaler=scaler, random_state=random_state+i))

    return results

def balance_resample(y_var:list = None, random_state:int = 42, sample_size:int = None, replace:bool = False):
    v, c = np.unique(y_var, return_counts=True)
    if sample_size == None:
        sample_size = c.min()
    else:
        if sample_size > c.min():
            print("Warning, sample_size greather than the minimum class counts. Truning sample_size =", str(c.min()))
            sample_size = c.min()

    idx = np.array([], dtype=int)
    for v_ in v:
        np.random.seed(random_state)
        idx = np.concatenate( ( idx, np.random.choice(np.where(y_var == v_)[0], size=sample_size, replace=replace) ))

    return idx

def set_balance_resample(y_var:list = None, n_set:int = 5, random_state:int = 42, sample_size:int = None, replace:bool = False):
    set_idx = []
    for n in range(n_set):
        set_idx.append( balance_resample(y_var, random_state+n, sample_size, replace) )

    return set_idx
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
