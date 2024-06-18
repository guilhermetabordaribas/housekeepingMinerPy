import warnings

import pandas as pd
import numpy as np
import anndata as ad
from scipy.stats import pearsonr, false_discovery_control, t, wald,

def tost(adata, conditions:str = 'ar', vars_list:str = [], cohens_d:float = .3):
