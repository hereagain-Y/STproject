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
from scipy.stats import expon
import scipy.stats as stats
from scipy.stats import poisson, norm,nbinom,gamma
import matplotlib.colors as mcolors
from functools import reduce

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
    
    
    
    def cauchy_combination_test(self, p_values):
        # Step 1: Convert p-values to Cauchy-distributed test statistics
        cauchy_stats = np.tan(np.pi * (p_values - 0.5))
        
        # Step 2: Calculate the combined test statistic (median of Cauchy stats)
        combined_stat = np.median(cauchy_stats)
        
        # Step 3: Calculate the combined p-value
        combined_p_value = stats.cauchy.cdf(combined_stat)
        
        # Adjust the p-value since the Cauchy distribution is symmetric around 0
        # We take the min of the calculated p-value and 1 minus that value to ensure the p-value is correctly oriented
        combined_p_value = min(combined_p_value, 1 - combined_p_value) * 2
        return combined_p_value

    def run_SVG(self, genes, grid_size, base, filter=False, method='default', dist='gamma'):
        '''
        ## input of function:  - count_data: DataFrame with gene expression counts.
        #                      - coord_data: DataFrame with x and y coordinates.
        # genes: list of genes to run the SVG test on
        # grid_size: size of the grid to bin the data into
        # base: base value for the grid size
        # filter: boolean to filter out outliers
        # method: method to use for the SVG test ('default logp ' or 'cauchy')
        # dist: distribution to use for the SVG test ('gamma', 'normal', 'poisson', 'nb', or 'all')
        Output: DataFrame with p-values for each gene based on the SVG test, colnames as genes, cauchy_{d} or logp_{d}. 
        '''
        
        y_bin_size = grid_size * base
        x_bin_size = grid_size * base

        x_min, x_max = self.coord['x'].min(), self.coord['x'].max()
        y_min, y_max = self.coord['y'].min(), self.coord['y'].max()

        self.coord['x_bin'] = np.floor((self.coord['x'] - x_min - x_bin_size / 2) / x_bin_size) + 1
        self.coord['y_bin'] = np.floor((self.coord['y'] - y_min - y_bin_size / 2) / y_bin_size) + 1
        self.coord['xy_bin'] = self.coord['x_bin'].astype(str) + "_" + self.coord['y_bin'].astype(str)

        df = pd.concat([self.coord['xy_bin'], self.count], axis=1)
        grouped_df = df.groupby(['xy_bin']).sum().reset_index()
        grouped_df.drop(['xy_bin'], axis=1, inplace=True)
        grouped_df = grouped_df.loc[grouped_df.sum(axis=1) != 0]

        lambda_values = grouped_df.mean()
        sd = grouped_df.std()
        Q1 = grouped_df.quantile(0.25)
        Q3 = grouped_df.quantile(0.75)
        IQR = Q3 - Q1
        read_counts = {col: grouped_df[col].value_counts() for col in grouped_df.columns}
        variance = sd**2

        p_value = pd.DataFrame(genes, columns=['gene'])
        distributions = ['gamma', 'normal', 'poisson', 'nb'] if dist == 'all' else [dist]

        for d in distributions:
            p_value[f'logp_{d}'] = [0] * len(genes)
            if method == 'cauchy':
                p_value[f'cauchy_{d}'] = [0] * len(genes)  # Ensure cauchy values are created for any 'cauchy' method

        for i in tqdm(range(len(genes))):
            gene = genes[i]
            logp_values = {d: 0 for d in distributions}
            cauchy_values = {d: [] for d in distributions}

            upper = Q3[gene] + 1.5 * IQR[gene]
            lower = Q1[gene] - 1.5 * IQR[gene]

            for count, bin_number in read_counts[gene].items():
                count = count if count != 0 else 0.1
                if not filter or (lower <= count <= upper):
                    for d in distributions:
                        if d == 'gamma':
                            k = lambda_values[gene]**2 / variance[gene]
                            theta = variance[gene] / lambda_values[gene]
                            pdf_value = gamma.pdf(count, a=k, scale=theta)
                        elif d == 'normal':
                            mu = lambda_values[gene]
                            sigma = np.sqrt(variance[gene])
                            pdf_value = norm.pdf(count, loc=mu, scale=sigma)
                        elif d == 'poisson':
                            mu = lambda_values[gene]
                            pdf_value = poisson.pmf(count, mu=mu)
                        elif d == 'nb':
                            mu = lambda_values[gene]
                            sigma_squared = variance[gene]
                            r = mu**2 / (sigma_squared - mu) if sigma_squared > mu else 1
                            p = mu / sigma_squared if sigma_squared > mu else 0.5
                            pdf_value = nbinom.pmf(count, r, p/(p+1)) 
                        logp_values[d] += bin_number * np.log(pdf_value + 1e-10)
                        if method == 'cauchy':
                            cauchy_values[d].extend([pdf_value] * bin_number)

            for d in distributions:
                p_value.loc[i, f'logp_{d}'] = logp_values[d]
                if method == 'cauchy':
                    p_value.loc[i, f'cauchy_{d}'] =self.cauchy_combination_test(np.array(cauchy_values[d]))

        # Select and return the relevant columns based on 'dist' and 'method' parameters
        relevant_cols = ['gene']
        if method == 'default':
            relevant_cols.extend([f'logp_{d}' for d in distributions])
        elif method == 'cauchy':
            relevant_cols.extend([f'cauchy_{d}' for d in distributions])

        p_value = p_value[relevant_cols]

        return p_value

    def run_comprehensive_SVG(self, genes, filter=False, method='default', dist='gamma'):
        results = []
        grid_sizes = [1, 2, 3, 4, 5, 6]
        # Run run_SVG for each grid size and append results to a list
        for size in grid_sizes:
            result = self.run_SVG(genes, size, size, filter, method, dist)
            result = result.rename(columns={col: f'{col}_{size}' for col in result.columns if col != 'gene'})
            results.append(result)
            
        comprehensive_result = pd.concat(results, axis=1)
        comprehensive_result = comprehensive_result.loc[:,~comprehensive_result.columns.duplicated()]


        # Calculate ranks for each grid size and the average rank
        for size in grid_sizes:
            comprehensive_result[f'rank_{size}'] = comprehensive_result[f'{method}_{dist}_{size}'].rank(method='min', ascending=False)
        
        rank_columns = [f'rank_{size}' for size in grid_sizes]
        comprehensive_result['average_rank'] = comprehensive_result[rank_columns].mean(axis=1)
        comprehensive_result['final_rank'] = comprehensive_result['average_rank'].rank(method='min').astype(int)

        # Select and order the columns appropriately
        result_columns = ['gene'] + [f'{method}_{dist}_{size}' for size in grid_sizes] +  ['final_rank']
        comprehensive_result = comprehensive_result[result_columns]
        comprehensive_result = comprehensive_result.sort_values('final_rank')
        return comprehensive_result
        
        
        
        

        
            
    def gene_expression_plot(self,gene,method,path):
        cmap = plt.get_cmap('plasma')
        #df = self.count[[gene,'x','y']]
        norm = mcolors.Normalize(vmin=self.count[gene].min(), vmax=self.count[gene].max())
        scatter = plt.scatter(self.coord['x'], self.coord['y'], c=self.count[gene], cmap=cmap, norm=norm, s=8)
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

