import warnings
import functools

import numpy as np
import pandas as pd
import anndata as ad

from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from sknetwork.clustering import Louvain
import umap

def create_groups(adata, layer:str = None, study_col:str = None,  scaler_object = None, nearestNeighbors_object = None, louvain_object = None, umap_object=None):
    """
    Create groups based on Nearest Neighbors and Louvain algorithm. It is usefull when the demographics and metadata are not available.
    The input must follow columns as features (genes) and rows as samples.

    Parameters
    ----------
    adata : Anndata
    layer : string, optional
        layer of AnnData object to be used to clustering.
        If layer is not informed, adata.X will be used.
    study_col : string, optional
        Columns in adata.obs to be used to stratify the clustering based on each study.
        If If layer is not informed, the data will be cluster for the entire dataset without stratification for different studies.
    scaler_object : scikit-learning scaler object
        Object to transform the data before clustering. By default no trasformation is applied.
        Example: sklearn.preprocessing.StandardScaler()
    nearestNeighbors_object : scikit-learning scaler object.
        scikit-learning object to calculate the neighbors. The default is the sklearn.neighbors.NearestNeighbors()
    louvain_object: sknetwork Object
        Louvain object to cluster the neighbors. The default is the sknetwork.clustering.Louvain()
    umap_object: UMAP Object
        Umap object to reduce the dimensionality only for 2D visualization purpose. The default is the umap.UMAP()

    Returns
    -------
    adata : AnnData object
        Return the same adata input with three other columns in adata.obs ('louvain_group','UMAP_1','UMAP_2')

    """
    idx = []
    lbl = []
    um1 = []
    um2 = []

    if study_col == None:
        if nearestNeighbors_object == None:
            nearestNeighbors_object = NearestNeighbors()

        if louvain_object == None:
            louvain_object = Louvain(random_state=42)

        if umap_object == None:
            umap_object = umap.UMAP()

        if layer == None:
            cl_X = adata.X
        else:
            cl_X = adata.layers[layer]

        if scaler_object != None:
            cl_X = scaler_object.fit_transform(cl_X)

        nearestNeighbors_object.fit(cl_X)
        labels = louvain_object.fit_predict(nearestNeighbors_object.kneighbors_graph())

        idx.extend(adata.obs.index.tolist())
        lbl.extend(labels.tolist())

        x_umap = umap_object.fit_transform(cl_X)
        um1.extend(x_umap[:,0].tolist())
        um2.extend(x_umap[:,1].tolist())
 
    else:
        for i,st_col in enumerate(adata.obs[study_col].unique()):
            if nearestNeighbors_object == None:
                nearestNeighbors_object = NearestNeighbors()

            if louvain_object == None:
                louvain_object = Louvain(random_state=42)

            if umap_object == None:
                umap_object = umap.UMAP()

            if layer == None:
                cl_X = adata[adata.obs[study_col]==st_col,:].X
            else:
                cl_X = adata[adata.obs[study_col]==st_col,:].layers[layer]

            if scaler_object != None:
                cl_X = scaler_object.fit_transform(cl_X)

            nearestNeighbors_object.fit(cl_X)
            labels = louvain_object.fit_predict(nearestNeighbors_object.kneighbors_graph())

            if i > 0:
                labels = labels + lbl_max + 1
            lbl_max = labels.max()

            idx.extend(adata[adata.obs[study_col]==st_col,:].obs.index.tolist())
            lbl.extend(labels.tolist())

            x_umap = umap_object.fit_transform(cl_X)
            um1.extend(x_umap[:,0].tolist())
            um2.extend(x_umap[:,1].tolist())

    adata.obs[['louvain_group','UMAP_1','UMAP_2']] = pd.DataFrame({'idx':idx, 'louvain_group':lbl, 'UMAP_1':um1, 'UMAP_2':um2}).set_index('idx').loc[adata.obs.index,:].values
    adata.obs['louvain_group'] = adata.obs['louvain_group'].astype(int)

    return adata

def log_transform(adata, layer:str = None, method:str = 'arcsinh'):
    """
    Apply log transformation to a expression table count adding a pseudocount (+1).

    Parameters
    ----------
    adata : Anndata
    layer : string, optional
        layer of AnnData object to be used to log transform.
        If layer is not informed, adata.X will be used.
    method : string
        Method of log transformation to avoid np.nan for log(0). It can be ['arcsinh', 'log1p'])
    Returns
    -------
    adata
        Return the adata with additional layer 'arcsinh' or 'log1p'.

    """

    if layer != None:
        X_ = adata.layers[layer]
    else:
        X_ = adata.X

    if method == 'arcsinh':
        adata.layers[method] = np.arcsinh(X_)
    elif method == 'log1p':
        adata.layers[method] = np.log1p(X_)
    else:
        raise Warning("method argument must be in ['arcsinh', 'log1p']. The method informed was " + str(method) + '.')

    return adata

def transform_exprs_RNAseq(X, trns_method:str = 'MRN' ):
    """
    Transform RNAseq counts data into MRN or TMM (DESeq2 and EdgeR respectively). This def used conorm algorithm to calculate the transformations.
    Important! The input must be the counts or pseudocounts integer counts, do not use TPM normalization as input.
    The input must follow columns as features (genes) and rows as samples.

    Parameters
    ----------
    X : np.array
        Integer count matrix
    trns_method : string, optional
        Type of method used. The options are MRN (DESeq2) or TMM (EdgeR).

    Returns
    -------
    np.array
        Return the X input transformed

    """
    if trns_method == 'MRN':
        return mrn(X.T).T
    elif trns_method == 'TMM':
        return tmm(X.T).T
    else:
        warnings.warn("Please, the norm_method must be 'MRN' or 'TMM'. Current norm_method = " + str(trns_method))

def transform_exprs_Microarray(X, trns_method:str = 'quantile' ):
    """
    Transform Microarray data into quantile or power transformation. This def used sklearn algorithm to calculate the transformations.
    Important! The input must be the counts or pseudocounts integer counts, do not use TPM normalization as input.
    The input must follow columns as features (genes) and rows as samples.

    Parameters
    ----------
    X : np.array
        Integer count matrix. Columns as features and rows as samples.
    norm_method : string or scikit-learning transform object
        Type of method used. The options are 'quantile' or 'power'. Or a scikit-learning sklearn.preprocessing object.

    Returns
    -------
    np.array
        Return the X input transformed

    """
    if isinstance(trns_method, str):
        if trns_method == 'quantile':
            return QuantileTransformer(output_distribution='normal').fit_transform(X.T).T
        elif trns_method == 'power':
            return PowerTransformer().fit_transform(X.T).T
        else:
            warnings.warn("Please, if you input a string, the norm_method must be 'quantile' or 'power'. Current norm_method = " + str(trns_method))
    else:
        return trns_method.fit_transform(X.T).T

def transform_exprs(adata, layer:str = None, groups_col:str = None, trns_dict:dict = None):
    """
    Transform expression data into MRN, TMM, quantile or power transformations.
    This information must be on a dict trns_dict, where the key is the group and the value the method, for example {0:'TMM', 1:'quantile'}.
    If the element in adata.obs trns_col column is not in ['MRN', 'TMM', 'quantile', 'power'], the subset will not be transformed.
    As well as the groups of interest must be related in adata.obs groups_col column to perform a grouped transformation.
    The input must follow columns as features (genes) and rows as samples.

    Parameters
    ----------
    adata : Anndata
    layer : string, optional
        layer of AnnData object to be used to transform.
        If layer is not informed, adata.X will be used.
    groups_col: string
        groups_col to perform the transformation independently. It must be a column at adata.obs annotations.
    trns_dict: dict
        trns_dict to perform the right transformation method to the subset. It must be a dictionary related to the adata.obs[groups_col].
        for example, if your groups are [0,1,2], the dict must be like {0:'TMM', 1:'TMM', 2:'quantile'}. For each group a respective method
        will be applied to tranformation.
        trns_dict values must be in ['MRN', 'TMM', 'quantile', 'power'], otherwise it will not be transformed.

    Returns
    -------
    adata
        Return the adata with additional layer 'trns'.

    """

    if groups_col == None:
        warnings.warn("No groups_col was informed. The data cannot be transformed without this info. Current groups_col is " + str(groups_col))
        return adata
    if trns_dict == None:
        warnings.warn("No trns_dict was informed. The data cannot be transformed without this info. Current trns_dict is " + str(trns_dict))
        return adata

    idx = []
    tr = []
    for gc in adata.obs[groups_col].unique():
        idx.extend( adata[adata.obs[groups_col]==gc,:].obs.index.tolist() )

        if layer != None:
            X_ = adata[adata.obs[groups_col]==gc,:].layers[layer]
        else:
            X_ = adata[adata.obs[groups_col]==gc,:].X

        if trns_dict[gc] in ['MRN', 'TMM']:
            tr.append( pd.DataFrame(transform_exprs_RNAseq(X_, trns_method = trns_dict[gc])) )
        elif trns_dict[gc] in ['quantile', 'power']:
            tr.append( pd.DataFrame(transform_exprs_Microarray(X_, trns_method = trns_dict[gc])) )
        else:
            tr.append( pd.DataFrame(X_) )
            warnings.warn("The trns_dict value for the group" + str(gc) + " is not one of ['MRN', 'TMM', 'quantile', 'power'] allowed values. This group will not be transformed. Current trns_dict['"+str(gc)+" = "+str(trns_dict[gc]))

    aux = pd.concat(tr)
    aux.index = idx
    adata.layers['trns_expr'] = aux.loc[adata.obs.index,:].values

    return adata




# All code above is from another package conorm https://gitlab.com/georgy.m/conorm
def tmm_norm_factors(data, trim_lfc=0.3, trim_mag=0.05, index_ref=None):
    """
    Compute Trimmed Means of M-values norm factors.

    Parameters
    ----------
    data : array_like
        Counts dataframe to normalize (rows are genes). Most often can be
        either pandas DataFrame or an numpy matrix.
    trim_lfc : float, optional
        Quantile cutoff for M_g (logfoldchanges). The default is 0.3.
    trim_mag : float, optional
        Quantile cutoff for A_g (log magnitude). The default is 0.05.
    index_ref : float, str, optional
        Reference index or column name to use as reference in the TMM
        algorithm. The default is None.

    Returns
    -------
    tmms : np.ndarray or pd.DataFrame
        Norm factors.

    """

    x = np.array(data, dtype=float).T
    lib_size = x.sum(axis=1)
    mask = x == 0
    if index_ref is None:
        x[:, np.all(mask, axis=0)] = np.nan
        p75 = np.nanpercentile(x, 75, axis=1)
        index_ref = np.argmin(abs(p75 - p75.mean()))
    mask[:, mask[index_ref]] = True
    x[mask] = np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        norm_x = x / lib_size[:, np.newaxis]
        logs = np.log2(norm_x)
        m_g =  logs - logs[index_ref]
        a_g = (logs + logs[index_ref]) / 2

        perc_m_g = np.nanquantile(m_g, [trim_lfc, 1 - trim_lfc], axis=1,
                                  method='nearest')[..., np.newaxis]
        perc_a_g = np.nanquantile(a_g, [trim_mag, 1 - trim_mag], axis=1,
                                  method='nearest')[..., np.newaxis]
        mask |= (m_g < perc_m_g[0]) | (m_g > perc_m_g[1])
        mask |= (a_g < perc_a_g[0]) | (a_g > perc_a_g[1])
        w_gk = (1 - norm_x) / x
        w_gk = 1 / (w_gk + w_gk[index_ref])
    w_gk[mask] = 0
    m_g[mask] = 0
    w_gk /= w_gk.sum(axis=1)[:, np.newaxis]
    tmms = np.sum(w_gk * m_g, axis=1)
    tmms -= tmms.mean()
    tmms = 2 ** tmms
    if type(data) is pd.DataFrame:
        tmms = pd.DataFrame(tmms, index=data.columns,
                            columns=['norm.factors'])
    return tmms


def tmm(data, trim_lfc=0.3, trim_mag=0.05, index_ref=None,
        return_norm_factors=False):
    """
    Normalize counts matrix by Trimmed Means of M-values (TMM).

    Parameters
    ----------
    data : array_like
        Counts dataframe to normalize (rows are genes). Most often can be
        either pandas DataFrame or an numpy matrix.
    trim_lfc : float, optional
        Quantile cutoff for M_g (logfoldchanges). The default is 0.3.
    trim_mag : float, optional
        Quantile cutoff for A_g (log magnitude). The default is 0.05.
    index_ref : float, str, optional
        Reference index or column name to use as reference in the TMM
        algorithm. The default is None.
    return_norm_factors : bool, optional
        If True, then norm factors are also returned. The default is False.

    Returns
    -------
    data : array_like
        Normalized data.

    """

    nf = tmm_norm_factors(data, trim_lfc=trim_lfc, trim_mag=trim_mag,
                          index_ref=index_ref)
    if return_norm_factors:
        return data / np.array(nf).flatten(), nf
    return data / np.array(nf).flatten()


def mrn_norm_factors(data):
    """
    Compute Median of Ratio norm factors.

    Parameters
    ----------
    data : array_like
        Counts dataframe to normalize (rows are genes). Most often can be
        either pandas DataFrame or an numpy matrix.

    Returns
    -------
    tmms : np.ndarray or pd.DataFrame
        Norm factors.

    """

    x = np.array(data, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        x /= x.mean(axis=1)[:, np.newaxis]
        x = np.log(x)
    x[~np.isfinite(x)] = np.nan
    nf = np.nanmedian(x, axis=0)
    nf -= nf.mean()
    nf = np.exp(nf)
    if type(data) is pd.DataFrame:
        nf = pd.DataFrame(nf, index=data.columns,
                          columns=['norm.factors'])
    return nf

def mrn(data, return_norm_factors=False):
    """
    Normalize counts matrix by Median of Ratios

    Parameters
    ----------
    data : array_like
        Counts dataframe to normalize (rows are genes). Most often can be
        either pandas DataFrame or an numpy matrix.
    return_norm_factors : bool, optional
        If True, then norm factors are also returned. The default is False.

    Returns
    -------
    data : array_like
        Normalized data.

    """

    nf = mrn_norm_factors(data)
    if return_norm_factors:
        return data / np.array(nf).flatten(), nf
    return data / np.array(nf).flatten()


def scaler(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        nf = kwargs.get('norm_factors')
        r = fun(*args, **kwargs)
        if nf is not None:
            if type(nf) is str:
                if nf == 'TMM':
                    nf = tmm_norm_factors(args[0])
                elif nf == 'MRN':
                    nf = mrn_norm_factors(args[0])
                else:
                    raise NotImplementedError(f'Unknown method {nf}.')
            r /= np.array(nf).flatten()
        return r
    return wrapper


@scaler
def total_count(matrix, norm_factors=None):
    """
    Total count normalization.


    Parameters
    ----------
    matrix : array_like
        Count data to normalize (rows are genes).
    norm_factors : array_like, optional
        Normalized factors to apply before doing CPM. Can also be a string
        'TMM' - then, norm factors are computed automatically. If None,
        then no norm factors are applied. The default is None.

    Returns
    -------
    array_like
        Normalized matrix.
    """
    return matrix / matrix.sum(axis=0)


def cpm(matrix, val=1e6, norm_factors=None):
    """
    Counts per million normalization.

    Total count normalization + multiplication by a million.
    Parameters
    ----------
    matrix : array_like
        Count data to normalize (rows are genes).
    val : float, optional
        Custom value to multiply afterwards. The default is 1e6.
    norm_factors : array_like, optional
        Normalized factors to apply before doing CPM. Can also be a string
        'TMM' - then, norm factors are computed automatically. If None,
        then no norm factors are applied. The default is None.

    Returns
    -------
    array_like
        Normalized matrix.

    """
    return total_count(matrix, norm_factors=norm_factors) * 1e6


@scaler
def percentile(matrix, p, norm_factors=None):
    """
    Percentile normalization

    Parameters
    ----------
    matrix : array_like
        Count data to normalize (rows are genes).
    p : float in range of [0,100]
        Percentile to compute, which must be between 0 and 100 inclusive.
    norm_factors : array_like
        Normalized factors to apply before doing CPM. Can also be a string
        'TMM' - then, norm factors are computed automatically. If None,
        then no norm factors are applied. The default is None.

    Returns
    -------
    array_like
        Normalized matrix.
    """
    return matrix / np.percentile(matrix[np.any(matrix > 0, axis=1)], p,
                                  axis=0, interpolation='nearest')


def quartile(matrix, q, norm_factors=None):
    """
    Quartile normalization.

    A wrapper around percentile normalization.
    Parameters
    ----------
    matrix : array_like
        Matrix or dataframe to normalize.
    q : str, int
        Quartile number or name. Can be either {"lower", "median", "upper"} or
       {1, 2, 3}.
    norm_factors : array_like
        Normalized factors to apply before doing CPM. Can also be a string
        'TMM' - then, norm factors are computed automatically. If None,
        then no norm factors are applied. The default is None.

    Returns
    -------
    array_like
        Normalized matrix.
    """
    d = {"upper": 75, "lower": 25, "median": 50, 3: 75, 1: 25, 2: 50}
    assert q in d, f'Unkown quartile name: {q}'
    return percentile(matrix, d[q], norm_factors=norm_factors)


@scaler
def rpk(matrix, length: str, norm_factors=None):
    """
    Reads per kilobase normalization.

    Counts are divided by a gene length.
    Parameters
    ----------
    matrix : array_like
        Count data to normalize (rows are genes).
    length : str
        Column name (if matrix is Pandas DataFrame), column index (if matrix
        is a numpy array) or an array, where gene lengths are stored.
    norm_factors : array_like, optional
        Normalized factors to apply before doing CPM. Can also be a string
        'TMM' - then, norm factors are computed automatically. If None,
        then no norm factors are applied. The default is None.

    Returns
    -------
    array_like
        Normalized matrix.

    """
    if type(length) is str:
        tlen = length
        length = np.array(matrix[length])
        matrix = matrix.drop(tlen, axis=1)
    elif type(length) is int:
        length = matrix[:, length]
    elif type(length) is pd.DataFrame:
        length = length.loc[matrix.index]
    length = np.array(length).reshape(-1, 1) / 1000
    return matrix / length

def rpkm(matrix, length: str, val=1e6, norm_factors=None):
    """
    Reads per kilobase normalization per million.

    Counts are divided by a gene length and multiplied by a million.
    Parameters
    ----------
    matrix : array_like
        Count data to normalize (rows are genes).
    length : str
        Column name (if matrix is Pandas DataFrame), column index (if matrix
        is a numpy array) or an array, where gene lengths are stored.
    val : float, optional
        Custom value to multiply afterwards. The default is 1e6.
    norm_factors : array_like, optional
        Normalized factors to apply before doing CPM. Can also be a string
        'TMM' - then, norm factors are computed automatically. If None,
        then no norm factors are applied. The default is None.

    Returns
    -------
    array_like
        Normalized matrix.

    """
    return rpk(matrix, length, norm_factors) * val


def getmm(matrix, length: str, trim_lfc=0.3, trim_mag=0.05, index_ref=None):
    """
    GeTMM normalization (RPK + TMM).

    Parameters
    ----------
    matrix : array_like
        Count data to normalize (rows are genes).
    length : str
        Column name (if matrix is Pandas DataFrame), column index (if matrix
        is a numpy array) or an array, where gene lengths are stored.
    trim_lfc : float, optional
        Quantile cutoff for M_g (logfoldchanges). The default is 0.3.
    trim_mag : float, optional
        Quantile cutoff for A_g (log magnitude). The default is 0.05.
    index_ref : float, str, optional
        Reference index or column name to use as reference in the TMM
        algorithm. The default is None.

    Returns
    -------
    array_like
        Normalized matrix.

    """
    matrix = rpk(matrix, length)
    return tmm(matrix, trim_lfc=trim_lfc, trim_mag=trim_mag,
               index_ref=index_ref)



def run_per_gene_position(gene_position: str, in_bam_file: str, path_out_res: str, ref_genome_file: str, path_reditools: str, reditools_options: str) -> None:
    """
    Run reditools2.0 via Python

    Parameters
    ----------
    gene_position : str
        coordinate of chromossome:start-end positions to search editing sites.
        Example: 'chr2:122147686-122153083'. One can get the coordinates from gene symbol using get_genes_positions function.
    in_bam_file : str
        full input BAM file path
    path_out_res: str
        output directory
    ref_genome_file: str
        full reference FASTA file path. Must the same used to build the aligned bam file.
    path_reditools: str
        full directory where reditools.py is installed. Usually is in similiar path '/../reditools2.0/src/cineca'
    reditools_options: str
        optional arguments to run reditools2. All the options are expalined in https://github.com/BioinfoUNIBA/REDItools2

    Returns
    -------
    None
        it doesn't return nothing, just run reditools2

    Example
    -------
        Using the toyfile from https://github.com/guilhermetabordaribas/a2iHelperPy

        >>> gene_position = 'chr2:122147686-122153083'
        >>> in_bam_file = '/.../sample1.sortedByCoord.out.bam'
        >>> path_out_res = '/.../out/'
        >>> ref_genome_file = '/.../GRCh38.p14.genome.fa'
        >>> path_reditools = '/.../reditools2.0/src/cineca/'
        >>> reditools_options = '--strict'
        >>> run_per_gene_position(gene_position, in_bam_file, path_out_res, ref_genome_file, path_reditools, reditools_options='--strict')
    """

    out_file = os.path.join( path_out_res, os.path.basename(in_bam_file)+'_'+gene_position.replace(':','_').replace('-','_')+'_RES.tsv' )
    cmd_list = ['python', 'reditools.py', '-f', in_bam_file, '-r', ref_genome_file, '-o', out_file, '-g', gene_position]
    if reditools_options:
        cmd_list += reditools_options.split(' ')
    subprocess.call(cmd_list, cwd=path_reditools, stdout=subprocess.PIPE)

def run_per_gene_position_list(genes_positions: list, in_bam_file: str, path_out_res: str, ref_genome_file: str, path_reditools: str, reditools_options: str, n_jobs:int = 4) -> None:
    """
    Run run_per_gene_position for a list of gense coordinates (genes_positions)

    Parameters
    ----------
    gene_position: list
        list of coordinates of chromossome:start-end positions to search editing sites.
        Example: ['chr2:122147686-122153083', 'chr18:60803848-60812646', 'chr6:65671590-65712326'].
        One can get the coordinates from gene symbol using get_genes_positions function.
    in_bam_file: str
        full input file BAM path
    path_out_res: str
        output directory
    ref_genome_file: str
        full reference FASTA file path. Must the same used to build the aligned bam file.
    path_reditools: str
        full directory where reditools.py is installed. Usually is in similiar path '/../reditools2.0/src/cineca'
    reditools_options: str
        optional arguments to run reditools2. All the options are expalined in https://github.com/BioinfoUNIBA/REDItools2
    n_jobs: int
        number of jobs in parallel

    Returns
    -------
    None
        it doesn't return nothing, just run reditools2 for a list o coordinates

    Example
    -------
        Using the toyfile from https://github.com/guilhermetabordaribas/a2iHelperPy

        >>> genes_positions = ['chr2:122147686-122153083', 'chr6:65671590-65712326', 'chr15:78191114-78206400']
        >>> in_bam_file = '/.../sample1.sortedByCoord.out.bam'
        >>> path_out_res = '/.../out/'
        >>> ref_genome_file = '/.../GRCh38.p14.genome.fa'
        >>> path_reditools = '/.../reditools2.0/src/cineca/'
        >>> reditools_options = '--strict'
        >>> run_per_gene_position_list(genes_positions, in_bam_file, path_out_res, ref_genome_file, path_reditools, reditools_options='--strict', n_jobs=4)
    """

    arguments_list = zip(genes_positions, itertools.repeat(in_bam_file), itertools.repeat(path_out_res), itertools.repeat(ref_genome_file), itertools.repeat(path_reditools), itertools.repeat(reditools_options))
    with Pool(processes=n_jobs) as p:
        p.starmap(run_per_gene_position, arguments_list)

def get_genes_positions(genes:list, path_ref_annotation:str, gzip_file:bool = False) -> list:
    """
    Return the coordinates of a gene symbol from a GTF file. It can be used as input to run_per_gene_position_list.

    Parameters
    ----------
    genes: list
        list of genes to get coordinates
    path_ref_annotation: str
        full reference GTF file path.

    Returns
    -------
        dict
            a dict of coordinates of each gene symbol (chr:start-end).

    Example
    -------
        Using the GTF file from gencode

        >>> get_genes_positions(['B2m', 'Apol1'], '/.../GRCh38.p14.genome.fa')
        ['chr2:122147686-122153083', 'chr18:60803848-60812646']
    """

    genes_positions_list = []
    gens_aux = []
    if genes:
        if gzip_file:
            # for g in genes:
            with gzip.open(path_ref_annotation,'r') as f_gtf:
                for line in f_gtf:
                    if not line.startswith('#'.encode()):
                        l = line.decode().split('\t')
                        dict_g = { i.split(' ')[0]:i.split(' ')[1] for i in [j.strip() for j in l[-1].replace(';\n','').replace('"','').split(';')] }
                        if (l[2]=='gene') and (dict_g['gene_name'] in genes):
                            g_list.append(l[0]+':'+l[3]+'-'+l[4])
                            gens_aux.append( dict_g['gene_name'] )
        else:
            # for g in genes:
            with open(path_ref_annotation,'r') as f_gtf:
                for line in f_gtf:
                    if not line.startswith('#'):
                        l = line.split('\t')
                        dict_g = { i.split(' ')[0]:i.split(' ')[1] for i in [j.strip() for j in l[-1].replace(';\n','').replace('"','').split(';')] }
                        if (l[2]=='gene') and (dict_g['gene_name'] in genes):
                            genes_positions_list.append(l[0]+':'+l[3]+'-'+l[4])
                            gens_aux.append( dict_g['gene_name'] )

    if not genes_positions_list:
        warnings.warn('*Returning empty list* -> Positions of genes were not found in the '+ path_ref_annotation+'. Please verify genes names or gtf file.')

    return dict(zip(gens_aux,genes_positions_list))

def get_utr_genes_positions(genes:list, path_ref_annotation:str, gzip_file:bool = False) -> list:
    """
    Return the coordinates of a gene symbol from a GTF file. It can be used as input to run_per_gene_position_list.

    Parameters
    ----------
    genes: list
        list of genes to get coordinates
    path_ref_annotation: str
        full reference GTF file path.

    Returns
    -------
        dict
            a dict of coordinates of each gene symbol (chr:start-end).

    Example
    -------
        Using the GTF file from gencode

        >>> get_genes_positions(['B2m', 'Apol1'], '/.../GRCh38.p14.genome.fa')
        ['chr2:122147686-122153083', 'chr18:60803848-60812646']
    """

    genes_positions_list = []
    gens_aux = []
    gzip_file = False
    get_gene = False
    chr = ''
    start_gene = ''
    end_gene = ''

    if genes:
        if gzip_file:
            # for g in genes:
            with gzip.open(path_ref_annotation,'r') as f_gtf:
                for line in f_gtf:
                    if not line.startswith('#'.encode()):
                        l = line.decode().split('\t')
                        dict_g = { i.split(' ')[0]:i.split(' ')[1] for i in [j.strip() for j in l[-1].replace(';\n','').replace('"','').split(';')] }
                        if (l[2]=='gene') and (dict_g['gene_name'] in genes):
                            chr = l[0]
                            start_gene = l[3]
                            end_gene = l[4]
                            get_gene = True
                            # gens_aux.append( dict_g['gene_name'] )
                        elif (get_gene) and (l[2]=='start_codon'):
                            start_codon = l[3]
                            gens_aux.append( dict_g['gene_name']+'_5UTR' )
                            if start_gene != start_codon:
                                genes_positions_list.append(chr+':'+start_gene+'-'+str(int(start_codon)-1))
                            else:
                                genes_positions_list.append(chr+':'+start_gene+'-'+start_codon)
                        elif (get_gene) and (l[2]=='stop_codon'):
                            stop_codon = l[4]
                            gens_aux.append( dict_g['gene_name']+'_3UTR' )
                            if end_gene != stop_codon:
                                genes_positions_list.append(chr+':'+str(int(stop_codon)+1)+'-'+end_gene)
                            else:
                                genes_positions_list.append(chr+':'+stop_codon+'-'+end_gene)
                            get_gene = False #control for get start and stop codon after gotten gene
        else:
            # for g in genes:
            with open(path_ref_annotation,'r') as f_gtf:
                for line in f_gtf:
                    if not line.startswith('#'):
                        l = line.split('\t')
                        dict_g = { i.split(' ')[0]:i.split(' ')[1] for i in [j.strip() for j in l[-1].replace(';\n','').replace('"','').split(';')] }
                        if (l[2]=='gene') and (dict_g['gene_name'] in genes):
                            chr = l[0]
                            start_gene = l[3]
                            end_gene = l[4]
                            get_gene = True
                            # gens_aux.append( dict_g['gene_name'] )
                        elif (get_gene) and (l[2]=='start_codon'):
                            start_codon = l[3]
                            gens_aux.append( dict_g['gene_name']+'_5UTR' )
                            if start_gene != start_codon:
                                genes_positions_list.append(chr+':'+start_gene+'-'+str(int(start_codon)-1))
                            else:
                                genes_positions_list.append(chr+':'+start_gene+'-'+start_codon)
                        elif (get_gene) and (l[2]=='stop_codon'):
                            stop_codon = l[4]
                            gens_aux.append( dict_g['gene_name']+'_3UTR' )
                            if end_gene != stop_codon:
                                genes_positions_list.append(chr+':'+str(int(stop_codon)+1)+'-'+end_gene)
                            else:
                                genes_positions_list.append(chr+':'+stop_codon+'-'+end_gene)
                            get_gene = False #control for get start and stop codon after gotten gene

    if not genes_positions_list:
        warnings.warn('*Returning empty list* -> Positions of genes were not found in the '+ path_ref_annotation+'. Please verify genes names or gtf file.')

    return dict(zip(gens_aux,genes_positions_list))

def indexing_ref(path_ref_genome):
    pass
