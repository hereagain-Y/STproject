# wrap code into class 

import h5py 
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import seaborn as sns
import os
import gzip
from scipy.io import mmread
from statsmodels.discrete.count_model import ZeroInflatedPoisson, Poisson
from statsmodels.tools import add_constant
from scipy.stats import norm
from scipy.stats import expon
import matplotlib.colors as mcolors

class SvgDetector:
    """
        Initializes the SvgDetector with count and coordinate data.
        
        Parameters:
        - count_data: DataFrame with gene expression counts.
        - coord_data: DataFrame with x and y coordinates.
    """
    def __init__(self,count,coord):
        
        self.count = count
        self.coord = coord
        self.filtered_count = None
        self.lambda_values = None
        self.sd = None
        self.read_counts = None
        self.p_value_df = None
        
    def filtergenes(self,expression_threshold,gene_det_in_cells, min_gene_per_cell):
    
        '''
        expression_threshold: sets the total expression threhold for each gene over all cells
        gene_det_in_cells: can be percentage or exact numeric number; minmum percentage of cells a gene is detected in,  minimum number of cells a gene is detected in.
        min_gene_per_cell: can be percentage or exact numeric number; minimum number /perentage of genes expressed per cell. 
        '''
    
        # sum expression_threshold #
        coordinates = self.count.iloc[:, -2:]
        col_sum = self.count.iloc[:,:-2].sum(axis=0)
        mask= col_sum>= expression_threshold
        expression_mask = col_sum.index[mask].tolist()
        
        
    # Filter the dataframe based on the columns to keep
        count= self.count[expression_mask]
        
        
        non_zeros = count >0
        
        # gene_det_in_cells for genes 
        if gene_det_in_cells <1: 
            
            total = len(count)
            # non -zero genes 
            min_cells_for_detection = total * gene_det_in_cells
            
        else: 
            min_cells_for_detection = gene_det_in_cells  
            
        filtered_genes = non_zeros.sum(axis = 0)>= min_cells_for_detection
        filter_count = count.loc[:, filtered_genes]
        
        # filter minumm # genes per cell  fillter out gene
        if min_gene_per_cell<1:
            total_genes = len(count.columns)-2
            min_genes_for_detection = total_genes * min_gene_per_cell 
        else: 
            min_genes_for_detection = min_gene_per_cell
        filtered_cells = non_zeros.sum(axis = 1)>= min_genes_for_detection 
        
        filter_count = filter_count.loc[filtered_cells,:]
        self.filter_count = pd.concat([filter_count, coordinates.loc[filtered_cells, :]], axis=1)
        return self.filter_count
    
    
    
    def run_SVG(self,count_data,gene_list,base):
    
        y_bin_size =120*base
        x_bin_size =140*base

        # Calculate the min and max values for x and y
        x_min, x_max = self.coord['x'].min(), self.coord['x'].max()
        y_min, y_max = self.coord['y'].min(), self.coord['y'].max()
        x_bins = int((x_max - x_min) / x_bin_size) + 1
        y_bins = int((y_max - y_min) / y_bin_size) + 1

        # Re-assign each data point to a bin
        self.coord['x_bin'] = (self.coord['x']-x_min - x_bin_size/2 )//x_bin_size +1
        self.coord['y_bin'] = (self.coord['y']-y_min - y_bin_size/2 )//y_bin_size +1

        self.coord['xy_bin'] = self.coord['x_bin'].astype(str) + "_" + self.coord['y_bin'].astype(str)

        self.coord['cnt'] = 1


        # Group by bins and cell type to aggregate the expression values
        grouped = self.coord.groupby(['x_bin', 'y_bin']).agg({'cnt':'count'}).reset_index()


        # Create a new dataframe with the binned data
        df = pd.concat([self.coord['xy_bin'] ,count_data], axis=1)
        grouped_df = df.groupby(['xy_bin']).sum().reset_index()
        
        grouped_df.drop(['xy_bin'], axis=1, inplace=True)
        # exclude blank bins: row sum is 0 
        grouped_df.loc[grouped_df.sum(axis=1)!=0]
        # calculate the mean of each gene
        lambda_values = grouped_df.mean()
        sd = grouped_df.std()
        self.read_counts = {col: grouped_df[col].value_counts() for col in grouped_df.columns}


        p_value = pd.DataFrame(gene_list,columns=['gene'])
        p_value['logp'] = [0]*len(gene_list)

        for i in  tqdm(range(len(gene_list))):
            gene = gene_list[i]
            logp = 0
            for bin_number, count in self.read_counts[gene].items():
                if count ==0:
                    count += 0.1
                up = lambda_values[gene]+ 3*sd[gene]
                down = max(lambda_values[gene]- 3*sd[gene],0)
            
                # fit an exponential distribution to the data
                if down<=count<=up:
                    rate_param = 1 / lambda_values[gene]  # Adjust according to your data's characteristics
                    logp += np.log(expon.pdf(count, scale=1/rate_param))
            p_value.loc[i,'logp'] = logp
            
        return p_value
        
    def gene_expression_plot(self,gene,method,path):
        cmap = plt.get_cmap('plasma')
        df = self.count[[gene,'x','y']]
        norm = mcolors.Normalize(vmin=self.count[gene].min(), vmax=self.count[gene].max())
        scatter = plt.scatter(df['x'], df['y'], c=df[gene], cmap=cmap, norm=norm, s=8)
        plt.colorbar(scatter, label='Gene Expression Level')

        # Formatting the plot
        plt.title(f'{gene}Expression Scatter Plot')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.savefig(f'{path}/{gene}.png',dpi=300)
        plt.show()
    
    def gene_count_hist(self,gene):
        plt.hist(self.count[gene], bins=30, edgecolor='black', alpha=0.7)

# Add titles and labels
        plt.title(f'Distribution of {gene} Counts')
        plt.xlabel('Gene Counts')
        plt.ylabel('Frequency')
        plt.show()

