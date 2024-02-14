#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:08:18 2024

@author: hereagain
"""
import sys
import h5py 
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import seaborn as sns
import os
import SpatialDE
import scipy
import gzip
from scipy.io import mmread
from statsmodels.discrete.count_model import ZeroInflatedPoisson, Poisson
from statsmodels.tools import add_constant
from scipy.stats import norm
import time 
from scipy.stats import expon
import matplotlib.colors as mcolors

sys.path.append('/Users/hereagain/Documents/Emory/Research/Steve Qin/spatial_transcriptomics/Code/Danwei')

from SVG_detector_danwei import SvgDetector

# load dat
count = pd.read_csv('/Users/hereagain/Documents/Emory/Research/Steve Qin/spatial_transcriptomics/dataset/try dataset/human_breast_cancer_1/human_breast_cancer_1.count.csv')

gene_id = pd.read_csv('//Users/hereagain/Documents/Emory/Research/Steve Qin/spatial_transcriptomics/dataset/try dataset/human_breast_cancer_1/human_breast_cancer_1.gene_name.csv')

coord = pd.read_csv('//Users/hereagain/Documents/Emory/Research/Steve Qin/spatial_transcriptomics/dataset/try dataset/human_breast_cancer_1/human_breast_cancer_1.loc.csv')
count['x'] = coord['V5']
count['y'] = coord['V6']
coord =coord.rename({'V5':'x','V6':'y'},axis=1)
columns_to_rename = [col for col in count.columns if col not in ['x', 'y']]

# Not necccesarry 
# check if duplicated gene id 
duplicates = gene_id['x'].duplicated()
# To display the duplicate values
duplicate_values = gene_id['x'][duplicates] 
dat=count.drop(columns=['x','y'])
mapping = dict(zip(dat.columns,gene_id['x']))
filtered_mapping = {k: mapping[k] for k in columns_to_rename if k in mapping}
count.rename(columns=filtered_mapping, inplace=True)
# drop duplicated columns 
duplicated_columns  = count.columns[count.columns.duplicated()]
new_count = count.drop(columns=duplicated_columns)  

plt.scatter(count['x'].tolist(),count['y'].tolist(),c='k',s=2)
plt.axis('equal')
plt.show()

x = coord ["x"].tolist()
y = coord ["y"].tolist()

# start analysis 
brain = SvgDetector(new_count, coord)
new_count_filter = brain.filtergenes(1,0.1,10)
new_count2 = new_count_filter.drop(columns=['x','y'])
gene_list = new_count2.columns.to_list() 

p_value1 = brain.run_SVG(new_count2,gene_list,base=1)
result= p_value1.sort_values(by='logp',ascending=True)
result['new_rank'] = result['logp'].rank(method='first', ascending=True) 
path= '/Users/hereagain/Desktop/Feb14'
brain.gene_expression_plot('COX6C','expon',path)
brain.gene_count_hist('COX6C')