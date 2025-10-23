```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
import sys

ROOT_DIR = Path.cwd().parent
DATA_DIR = ROOT_DIR / 'data'
PLOT_DIR = ROOT_DIR / 'plots'

sys.path.append(str(ROOT_DIR))

from src.utils import (
    mean_mixed_corr_matrix, 
    plot_corr_heatmap, 
    WeightedStandardScaler,
    WeightedPCA,
    create_euclidean_distance_matrix,
    plot_dendrogram,
    get_clusters_from_dendrogram,
    print_cluster_summary,
    visualize_cluster_graph
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

years = [2008, 2012, 2016, 2020]
df = pd.read_csv(DATA_DIR / 'processed_data/probability_dataset.csv')

# For EDA, we will focus on the training data (2008, 2012, 2016)
df = df[df['year'].isin([2008, 2012, 2016])].reset_index(drop=True)

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9168 entries, 0 to 9167
    Data columns (total 62 columns):
     #   Column                              Non-Null Count  Dtype  
    ---  ------                              --------------  -----  
     0   gisjoin                             9168 non-null   object 
     1   year                                9168 non-null   int64  
     2   state                               9168 non-null   object 
     3   county                              9168 non-null   object 
     4   P(C)                                9168 non-null   float64
     5   P(area_in_C)                        9168 non-null   float64
     6   median_household_income             9168 non-null   int64  
     7   per_capita_income                   9168 non-null   int64  
     8   population_density                  9168 non-null   float64
     9   P(income_10k_to_15k|house_in_C)     9168 non-null   float64
     10  P(income_15k_to_25k|house_in_C)     9168 non-null   float64
     11  P(income_25k_plus|house_in_C)       9168 non-null   float64
     12  P(income_less_than_10k|house_in_C)  9168 non-null   float64
     13  P(poverty|C)                        9168 non-null   float64
     14  P(hispanic|C)                       9168 non-null   float64
     15  P(nativity_native|C)                9168 non-null   float64
     16  P(labor_16_plus_employed|C)         9168 non-null   float64
     17  P(labor_16_plus_unemployed|C)       9168 non-null   float64
     18  P(labor_16_plus_armed_forces|C)     9168 non-null   float64
     19  P(labor_16_plus_not_in_force|C)     9168 non-null   float64
     20  P(white_total|C)                    9168 non-null   float64
     21  P(black_total|C)                    9168 non-null   float64
     22  P(asian_total|C)                    9168 non-null   float64
     23  P(aian_total|C)                     9168 non-null   float64
     24  P(nhpi_total|C)                     9168 non-null   float64
     25  P(other_total|C)                    9168 non-null   float64
     26  P(multi_total|C)                    9168 non-null   float64
     27  P(male_age_low_edu_low|C)           9168 non-null   float64
     28  P(female_age_low_edu_low|C)         9168 non-null   float64
     29  P(male_age_low_edu_mid|C)           9168 non-null   float64
     30  P(female_age_low_edu_mid|C)         9168 non-null   float64
     31  P(male_age_low_edu_high|C)          9168 non-null   float64
     32  P(female_age_low_edu_high|C)        9168 non-null   float64
     33  P(male_age_low_edu_very_high|C)     9168 non-null   float64
     34  P(female_age_low_edu_very_high|C)   9168 non-null   float64
     35  P(male_age_mid_edu_low|C)           9168 non-null   float64
     36  P(female_age_mid_edu_low|C)         9168 non-null   float64
     37  P(male_age_mid_edu_mid|C)           9168 non-null   float64
     38  P(female_age_mid_edu_mid|C)         9168 non-null   float64
     39  P(male_age_mid_edu_high|C)          9168 non-null   float64
     40  P(female_age_mid_edu_high|C)        9168 non-null   float64
     41  P(male_age_mid_edu_very_high|C)     9168 non-null   float64
     42  P(female_age_mid_edu_very_high|C)   9168 non-null   float64
     43  P(male_age_high_edu_low|C)          9168 non-null   float64
     44  P(female_age_high_edu_low|C)        9168 non-null   float64
     45  P(male_age_high_edu_mid|C)          9168 non-null   float64
     46  P(female_age_high_edu_mid|C)        9168 non-null   float64
     47  P(male_age_high_edu_high|C)         9168 non-null   float64
     48  P(female_age_high_edu_high|C)       9168 non-null   float64
     49  P(male_age_high_edu_very_high|C)    9168 non-null   float64
     50  P(female_age_high_edu_very_high|C)  9168 non-null   float64
     51  P(marital_divorced|C)               9168 non-null   float64
     52  P(marital_single|C)                 9168 non-null   float64
     53  P(marital_married|C)                9168 non-null   float64
     54  P(marital_separated|C)              9168 non-null   float64
     55  P(marital_widowed|C)                9168 non-null   float64
     56  racial_diversity                    9168 non-null   float64
     57  P(democrat_voter|C)                 9168 non-null   float64
     58  P(republican_voter|C)               9168 non-null   float64
     59  P(other_voter|C)                    9168 non-null   float64
     60  P(non_voter|C)                      9168 non-null   float64
     61  log_odds_dem_rep                    9168 non-null   float64
    dtypes: float64(56), int64(3), object(3)
    memory usage: 4.3+ MB


$$r_{XY} = \frac{ \sum_{i=1}^{n} w_i (x_i - \bar{x}_w) (y_i - \bar{y}_w) }{ \sqrt{ \sum_{i=1}^{n} w_i (x_i - \bar{x}_w)^2 } \cdot \sqrt{ \sum_{i=1}^{n} w_i (y_i - \bar{y}_w)^2 } }$$


```python
id_cols = df.columns[:4].tolist()
target_cols = df.columns[-5:].tolist()
feature_cols = df.columns[4:-5].tolist()
all_cols = feature_cols + target_cols

# Compute mean mixed correlation matrices for the 3 train years [2008, 2012, 2016]
unweighted_avg_corr_matrix = mean_mixed_corr_matrix(df, columns=all_cols, weight_col=None, years=years[:-1])
weighted_avg_corr_matrix = mean_mixed_corr_matrix(df, columns=all_cols, weight_col='P(C)', years=years[:-1])

# Compute the difference
corr_diff = weighted_avg_corr_matrix - unweighted_avg_corr_matrix

# truncate to 3 decimal places
unweighted_avg_corr_matrix = unweighted_avg_corr_matrix.round(3)
weighted_avg_corr_matrix = weighted_avg_corr_matrix.round(3)
corr_diff = corr_diff.round(3)

# save to csv
CORR_DIR = DATA_DIR / 'correlation_matrices'
CORR_DIR.mkdir(exist_ok=True)
unweighted_avg_corr_matrix.to_csv(CORR_DIR / 'corr_matrix_unweighted.csv', index=True)
weighted_avg_corr_matrix.to_csv(CORR_DIR / 'corr_matrix_weighted.csv', index=True)
corr_diff.to_csv(CORR_DIR / 'corr_matrix_difference.csv', index=True)
```


```python
plot_corr_heatmap(
    unweighted_avg_corr_matrix,
    title='Unweighted Average Correlation Matrix (Pearson in bottom left, Spearman in top right)',
    save=True
)
```


    
![png](exploratory-data-analysis_files/exploratory-data-analysis_3_0.png)
    



```python
plot_corr_heatmap(
    weighted_avg_corr_matrix,
    title='Weighted Average Correlation Matrix (Pearson in bottom left, Spearman in top right)',
    save=True
)
```


    
![png](exploratory-data-analysis_files/exploratory-data-analysis_4_0.png)
    



```python
# Compute the symmetric avg weighted corr matrix
weighted_corr_symmetric = (weighted_avg_corr_matrix + weighted_avg_corr_matrix.T) / 2

# Compute distances
distance_matrix = create_euclidean_distance_matrix(weighted_corr_symmetric).round(3)
```


```python
fig, ax, linkage_mat = plot_dendrogram(
    distance_matrix, 
    title="Feature Clustering Based on Euclidean Distances",
    save=True
)
```


    
![png](exploratory-data-analysis_files/exploratory-data-analysis_6_0.png)
    



```python
# Extract clusters and print summary
clusters = get_clusters_from_dendrogram(
    corr_matrix=weighted_avg_corr_matrix, 
    linkage_matrix=linkage_mat,
    criterion='distance',
    t=2.5)
print_cluster_summary(clusters, weighted_avg_corr_matrix)
```

    Found 14 clusters:
    
    Cluster 7 (8 features):
      Features: median_household_income, per_capita_income, P(income_25k_plus|house_in_C), P(labor_16_plus_employed|C), P(male_age_mid_edu_high|C), P(female_age_mid_edu_high|C), P(male_age_mid_edu_very_high|C), P(female_age_mid_edu_very_high|C)
      Min internal absolute correlation: 0.466
      Average internal absolute correlation: 0.750
      Max internal absolute correlation: 0.957
    
    Cluster 10 (6 features):
      Features: P(C), population_density, P(asian_total|C), P(marital_single|C), racial_diversity, log_odds_dem_rep
      Min internal absolute correlation: 0.510
      Average internal absolute correlation: 0.631
      Max internal absolute correlation: 0.862
    
    Cluster 5 (6 features):
      Features: P(income_10k_to_15k|house_in_C), P(income_15k_to_25k|house_in_C), P(income_less_than_10k|house_in_C), P(poverty|C), P(male_age_mid_edu_low|C), P(female_age_mid_edu_low|C)
      Min internal absolute correlation: 0.578
      Average internal absolute correlation: 0.721
      Max internal absolute correlation: 0.884
    
    Cluster 8 (5 features):
      Features: P(male_age_low_edu_high|C), P(female_age_low_edu_high|C), P(male_age_low_edu_very_high|C), P(female_age_low_edu_very_high|C), P(democrat_voter|C)
      Min internal absolute correlation: 0.471
      Average internal absolute correlation: 0.699
      Max internal absolute correlation: 0.906
    
    Cluster 3 (5 features):
      Features: P(male_age_mid_edu_mid|C), P(female_age_mid_edu_mid|C), P(male_age_high_edu_mid|C), P(female_age_high_edu_mid|C), P(marital_divorced|C)
      Min internal absolute correlation: 0.471
      Average internal absolute correlation: 0.641
      Max internal absolute correlation: 0.845
    
    Cluster 14 (4 features):
      Features: P(area_in_C), P(aian_total|C), P(male_age_low_edu_mid|C), P(female_age_low_edu_mid|C)
      Min internal absolute correlation: 0.246
      Average internal absolute correlation: 0.427
      Max internal absolute correlation: 0.739
    
    Cluster 4 (4 features):
      Features: P(labor_16_plus_not_in_force|C), P(male_age_high_edu_low|C), P(female_age_high_edu_low|C), P(marital_widowed|C)
      Min internal absolute correlation: 0.683
      Average internal absolute correlation: 0.762
      Max internal absolute correlation: 0.874
    
    Cluster 6 (4 features):
      Features: P(male_age_high_edu_high|C), P(female_age_high_edu_high|C), P(male_age_high_edu_very_high|C), P(female_age_high_edu_very_high|C)
      Min internal absolute correlation: 0.650
      Average internal absolute correlation: 0.769
      Max internal absolute correlation: 0.832
    
    Cluster 2 (3 features):
      Features: P(nativity_native|C), P(white_total|C), P(republican_voter|C)
      Min internal absolute correlation: 0.596
      Average internal absolute correlation: 0.628
      Max internal absolute correlation: 0.649
    
    Cluster 12 (3 features):
      Features: P(labor_16_plus_unemployed|C), P(black_total|C), P(marital_separated|C)
      Min internal absolute correlation: 0.397
      Average internal absolute correlation: 0.469
      Max internal absolute correlation: 0.529
    
    Cluster 13 (3 features):
      Features: P(labor_16_plus_armed_forces|C), P(nhpi_total|C), P(multi_total|C)
      Min internal absolute correlation: 0.324
      Average internal absolute correlation: 0.411
      Max internal absolute correlation: 0.579
    
    Cluster 11 (3 features):
      Features: P(male_age_low_edu_low|C), P(female_age_low_edu_low|C), P(non_voter|C)
      Min internal absolute correlation: 0.591
      Average internal absolute correlation: 0.659
      Max internal absolute correlation: 0.774
    
    Cluster 9 (2 features):
      Features: P(hispanic|C), P(other_total|C)
      Min internal absolute correlation: 0.920
      Average internal absolute correlation: 0.920
      Max internal absolute correlation: 0.920
    
    Cluster 1 (2 features):
      Features: P(marital_married|C), P(other_voter|C)
      Min internal absolute correlation: 0.250
      Average internal absolute correlation: 0.250
      Max internal absolute correlation: 0.250
    



```python
# Visualize top 8 clusters
top_8_clusters = dict(list(clusters.items())[:8])
visualize_cluster_graph(
    top_8_clusters,
    weighted_avg_corr_matrix,
    unique_node_colors=True,
    show_node_labels=False,
    show_edge_labels=True,
    add_legend=True,
    save=True
)
```


    
![png](exploratory-data-analysis_files/exploratory-data-analysis_8_0.png)
    



    
![png](exploratory-data-analysis_files/exploratory-data-analysis_8_1.png)
    



    
![png](exploratory-data-analysis_files/exploratory-data-analysis_8_2.png)
    



    
![png](exploratory-data-analysis_files/exploratory-data-analysis_8_3.png)
    



    
![png](exploratory-data-analysis_files/exploratory-data-analysis_8_4.png)
    



    
![png](exploratory-data-analysis_files/exploratory-data-analysis_8_5.png)
    



    
![png](exploratory-data-analysis_files/exploratory-data-analysis_8_6.png)
    



    
![png](exploratory-data-analysis_files/exploratory-data-analysis_8_7.png)
    



```python
# Add a column for winner of each county
df["winner"] = 0
df.loc[df["P(democrat_voter|C)"] > df["P(republican_voter|C)"],"winner"] = 1

# Create dictionary mapping demographic types to feature lists
demographic_types = {
    "id": id_cols,
    "general": ["P(C)", "P(area_in_C)", "population_density"],
    "household_income": ["median_household_income", "per_capita_income", "P(poverty|C)"] + [col for col in df.columns if "income_" in col],
    "ethnicity": ["P(hispanic|C)", "P(nativity_native|C)"],
    "labor": [col for col in df.columns if "labor_" in col],
    "race": [col for col in df.columns if "_total|C)" in col] + ["racial_diversity"],
    "sex_age_edu": [col for col in df.columns if "edu_" in col],
    "marital": [col for col in df.columns if "marital_" in col],
    "target": [col for col in df.columns if "voter|C)" in col] + ["log_odds_dem_rep", "winner"]
}

demographic_types
```




    {'id': ['gisjoin', 'year', 'state', 'county'],
     'general': ['P(C)', 'P(area_in_C)', 'population_density'],
     'household_income': ['median_household_income',
      'per_capita_income',
      'P(poverty|C)',
      'P(income_10k_to_15k|house_in_C)',
      'P(income_15k_to_25k|house_in_C)',
      'P(income_25k_plus|house_in_C)',
      'P(income_less_than_10k|house_in_C)'],
     'ethnicity': ['P(hispanic|C)', 'P(nativity_native|C)'],
     'labor': ['P(labor_16_plus_employed|C)',
      'P(labor_16_plus_unemployed|C)',
      'P(labor_16_plus_armed_forces|C)',
      'P(labor_16_plus_not_in_force|C)'],
     'race': ['P(white_total|C)',
      'P(black_total|C)',
      'P(asian_total|C)',
      'P(aian_total|C)',
      'P(nhpi_total|C)',
      'P(other_total|C)',
      'P(multi_total|C)',
      'racial_diversity'],
     'sex_age_edu': ['P(male_age_low_edu_low|C)',
      'P(female_age_low_edu_low|C)',
      'P(male_age_low_edu_mid|C)',
      'P(female_age_low_edu_mid|C)',
      'P(male_age_low_edu_high|C)',
      'P(female_age_low_edu_high|C)',
      'P(male_age_low_edu_very_high|C)',
      'P(female_age_low_edu_very_high|C)',
      'P(male_age_mid_edu_low|C)',
      'P(female_age_mid_edu_low|C)',
      'P(male_age_mid_edu_mid|C)',
      'P(female_age_mid_edu_mid|C)',
      'P(male_age_mid_edu_high|C)',
      'P(female_age_mid_edu_high|C)',
      'P(male_age_mid_edu_very_high|C)',
      'P(female_age_mid_edu_very_high|C)',
      'P(male_age_high_edu_low|C)',
      'P(female_age_high_edu_low|C)',
      'P(male_age_high_edu_mid|C)',
      'P(female_age_high_edu_mid|C)',
      'P(male_age_high_edu_high|C)',
      'P(female_age_high_edu_high|C)',
      'P(male_age_high_edu_very_high|C)',
      'P(female_age_high_edu_very_high|C)'],
     'marital': ['P(marital_divorced|C)',
      'P(marital_single|C)',
      'P(marital_married|C)',
      'P(marital_separated|C)',
      'P(marital_widowed|C)'],
     'target': ['P(democrat_voter|C)',
      'P(republican_voter|C)',
      'P(other_voter|C)',
      'P(non_voter|C)',
      'log_odds_dem_rep',
      'winner']}




```python
# Create dictionaries for top 2 principal components (unweighted and weighted)

demo_types = [k for k in demographic_types.keys() if k not in ['id', 'target']]
features = [col for col in df.columns if col not in id_cols + target_cols]
weights = df['P(C)'].values
X = df[features].values

# Unweighted standardization
scaler_unw = StandardScaler()
X_unw_scaled = scaler_unw.fit_transform(X)
scaled_df_unw = pd.DataFrame(X_unw_scaled, columns=features, index=df.index)

# Weighted standardization
scaler_wtd = WeightedStandardScaler()
X_wtd_scaled = scaler_wtd.fit_transform(X, weights)
scaled_df_wtd = pd.DataFrame(X_wtd_scaled, columns=features, index=df.index)

# Initialize dictionaries
unweighted_pcs = {}
weighted_pcs = {}

for demo_type in demo_types:
    features_demo = demographic_types[demo_type]
    
    # Unweighted: use pre-standardized data and sklearn PCA
    X_demo_unw = scaled_df_unw[features_demo].values
    pca_unw = PCA(n_components=2)
    pca_unw.fit(X_demo_unw)
    unweighted_pcs[demo_type] = pca_unw.components_  # Shape: (2, n_features)
    
    # Weighted: use pre-standardized data and custom WeightedPCA with original weights
    X_demo_wtd = scaled_df_wtd[features_demo].values
    pca_wtd = WeightedPCA(n_components=2)
    pca_wtd.fit(X_demo_wtd, weights)
    weighted_pcs[demo_type] = pca_wtd.components_  # Shape: (2, n_features)
```


```python


# Define colormap for log_odds_dem_rep: red for negative, blue for positive
cmap = plt.cm.RdBu  # Red to blue
norm = mcolors.Normalize(vmin=df['log_odds_dem_rep'].min(), vmax=df['log_odds_dem_rep'].max())

# Create subplots: 7 rows x 2 columns (one row per demographic type, unweighted vs weighted)
fig, axes = plt.subplots(7, 2, figsize=(15, 35))  # Taller figure for 7 rows

for i, demo_type in enumerate(demo_types):
    features = demographic_types[demo_type]
    log_odds = df['log_odds_dem_rep'].values
    
    X_demo_unw = scaled_df_unw[features].values
    X_demo_wtd = scaled_df_wtd[features].values
    
    X_pca_unweighted = X_demo_unw @ unweighted_pcs[demo_type].T
    X_pca_weighted = X_demo_wtd @ weighted_pcs[demo_type].T

    # Plot Unweighted (left)
    ax = axes[i, 0]
    sc = ax.scatter(X_pca_unweighted[:, 0], X_pca_unweighted[:, 1],
                    c=log_odds, cmap=cmap, norm=norm, alpha=0.6, s=20)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'{demo_type.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3)

    # Plot Weighted (right)
    ax = axes[i, 1]
    sc = ax.scatter(X_pca_weighted[:, 0], X_pca_weighted[:, 1],
                    c=log_odds, cmap=cmap, norm=norm, alpha=0.6, s=20)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'{demo_type.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3)

# Add colorbar
cbar = fig.colorbar(sc, ax=axes, orientation='horizontal', fraction=0.02, pad=0.04)
cbar.set_label('Log Odds Democrat vs Republican')

plt.suptitle('Scatterplot of demographic Unweighted (left) and Weighted (right) Principal Components, colored by Log Odds of Dem over Repub', fontsize=18, fontweight='bold', y=0.91)

plt.show()

# Save the figure
fig.savefig(PLOT_DIR / 'demographic_pca_scatter_unweighted_weighted.png', bbox_inches='tight', dpi=300)
```


    
![png](exploratory-data-analysis_files/exploratory-data-analysis_11_0.png)
    



```python
# Modified: Analyze PC1 scatterplots for pairs of demographic types (Unweighted vs Weighted)
# Coloring by log_odds_dem_rep with red (negative) to blue (positive) gradient

import itertools
import matplotlib.colors as mcolors

# Get all pairs of demographic types excluding id and target
demo_types = [k for k in demographic_types.keys() if k not in ['id', 'target']]
pairs = list(itertools.combinations(demo_types, 2))

# Define colormap for log_odds_dem_rep: red for negative, blue for positive
cmap = plt.cm.RdBu  # Red to blue
norm = mcolors.Normalize(vmin=df['log_odds_dem_rep'].min(), vmax=df['log_odds_dem_rep'].max())

# Create subplots: one row per pair, unweighted vs weighted
fig, axes = plt.subplots(len(pairs), 2, figsize=(15, 5 * len(pairs)))

log_odds = df['log_odds_dem_rep'].values

for i, (A, B) in enumerate(pairs):
    # Get pre-computed PC1 for each demographic type
    features_A = demographic_types[A]
    features_B = demographic_types[B]
    
    # Transform data using pre-computed principal components (PC1 only)
    X_demo_A_unw = scaled_df_unw[features_A].values
    X_demo_B_unw = scaled_df_unw[features_B].values
    X_demo_A_wtd = scaled_df_wtd[features_A].values
    X_demo_B_wtd = scaled_df_wtd[features_B].values
    
    # Get PC1 for unweighted
    pc1_A_unw = X_demo_A_unw @ unweighted_pcs[A][0, :]  # First component only
    pc1_B_unw = X_demo_B_unw @ unweighted_pcs[B][0, :]
    
    # Get PC1 for weighted
    pc1_A_wtd = X_demo_A_wtd @ weighted_pcs[A][0, :]
    pc1_B_wtd = X_demo_B_wtd @ weighted_pcs[B][0, :]
    
    # Plot Unweighted (left)
    ax = axes[i, 0]
    sc = ax.scatter(pc1_A_unw, pc1_B_unw,
                    c=log_odds, cmap=cmap, norm=norm, alpha=0.6, s=20)
    ax.set_xlabel(f'PC1 {A.replace("_", " ").title()}')
    ax.set_ylabel(f'PC1 {B.replace("_", " ").title()}')
    ax.grid(True, alpha=0.15)
    
    # Plot Weighted (right)
    ax = axes[i, 1]
    sc = ax.scatter(pc1_A_wtd, pc1_B_wtd,
                    c=log_odds, cmap=cmap, norm=norm, alpha=0.6, s=20)
    ax.set_xlabel(f'PC1 {A.replace("_", " ").title()}')
    ax.set_ylabel(f'PC1 {B.replace("_", " ").title()}')
    ax.grid(True, alpha=0.15)

# Add colorbar
cbar = fig.colorbar(sc, ax=axes, orientation='horizontal', fraction=0.02, pad=0.04)
cbar.set_label('Log Odds Democrat vs Republican')

plt.suptitle('PC1 Scatterplots for Demographic Pairings: Unweighted (left) vs Weighted (right), colored by Log Odds of Dem over Repub', fontsize=18, fontweight='bold', y=0.89)

plt.show()

# save the figure
fig.savefig(PLOT_DIR / 'demographic_pairs_pc1_scatter_unweighted_weighted.png', bbox_inches='tight', dpi=300)
```


    
![png](exploratory-data-analysis_files/exploratory-data-analysis_12_0.png)
    



```python
# Compute top 2 principal components for each cluster (instead of demographic type)

# First, filter out clusters with only 1 feature or handle 2-feature clusters specially
cluster_pcs = {}
cluster_features = {}
cluster_methods = {}  # Track whether we used PCA or raw features

for cluster_id, features in clusters.items():
    features = [f for f in features if f not in target_cols] 
    if len(features) == 1:
        print(f"Skipping Cluster {cluster_id} - only 1 feature: {features[0]}")
        continue
    elif len(features) == 2:
        print(f"Using raw features for Cluster {cluster_id} - exactly 2 features: {features}")
        cluster_features[cluster_id] = features
        cluster_methods[cluster_id] = 'raw'
        cluster_pcs[cluster_id] = None  # No PCA needed
    else:
        print(f"Computing PCA for Cluster {cluster_id} - {len(features)} features: {features}")
        cluster_features[cluster_id] = features
        cluster_methods[cluster_id] = 'pca'
        
        # Unweighted PCA
        X_cluster_unw = scaled_df_unw[features].values
        pca_unw = PCA(n_components=2)
        pca_unw.fit(X_cluster_unw)
        
        # Weighted PCA
        X_cluster_wtd = scaled_df_wtd[features].values
        pca_wtd = WeightedPCA(n_components=2)
        pca_wtd.fit(X_cluster_wtd, weights)
        
        cluster_pcs[cluster_id] = {
            'unweighted': pca_unw.components_,  # Shape: (2, n_features)
            'weighted': pca_wtd.components_     # Shape: (2, n_features)
        }

print(f"\nProcessing {len(cluster_features)} clusters (skipped {len(clusters) - len(cluster_features)} single-feature clusters)")
```

    Computing PCA for Cluster 7 - 8 features: ['median_household_income', 'per_capita_income', 'P(income_25k_plus|house_in_C)', 'P(labor_16_plus_employed|C)', 'P(male_age_mid_edu_high|C)', 'P(female_age_mid_edu_high|C)', 'P(male_age_mid_edu_very_high|C)', 'P(female_age_mid_edu_very_high|C)']
    Computing PCA for Cluster 10 - 5 features: ['P(C)', 'population_density', 'P(asian_total|C)', 'P(marital_single|C)', 'racial_diversity']
    Computing PCA for Cluster 5 - 6 features: ['P(income_10k_to_15k|house_in_C)', 'P(income_15k_to_25k|house_in_C)', 'P(income_less_than_10k|house_in_C)', 'P(poverty|C)', 'P(male_age_mid_edu_low|C)', 'P(female_age_mid_edu_low|C)']
    Computing PCA for Cluster 8 - 4 features: ['P(male_age_low_edu_high|C)', 'P(female_age_low_edu_high|C)', 'P(male_age_low_edu_very_high|C)', 'P(female_age_low_edu_very_high|C)']
    Computing PCA for Cluster 3 - 5 features: ['P(male_age_mid_edu_mid|C)', 'P(female_age_mid_edu_mid|C)', 'P(male_age_high_edu_mid|C)', 'P(female_age_high_edu_mid|C)', 'P(marital_divorced|C)']
    Computing PCA for Cluster 14 - 4 features: ['P(area_in_C)', 'P(aian_total|C)', 'P(male_age_low_edu_mid|C)', 'P(female_age_low_edu_mid|C)']
    Computing PCA for Cluster 4 - 4 features: ['P(labor_16_plus_not_in_force|C)', 'P(male_age_high_edu_low|C)', 'P(female_age_high_edu_low|C)', 'P(marital_widowed|C)']
    Computing PCA for Cluster 6 - 4 features: ['P(male_age_high_edu_high|C)', 'P(female_age_high_edu_high|C)', 'P(male_age_high_edu_very_high|C)', 'P(female_age_high_edu_very_high|C)']
    Using raw features for Cluster 2 - exactly 2 features: ['P(nativity_native|C)', 'P(white_total|C)']
    Computing PCA for Cluster 12 - 3 features: ['P(labor_16_plus_unemployed|C)', 'P(black_total|C)', 'P(marital_separated|C)']
    Computing PCA for Cluster 13 - 3 features: ['P(labor_16_plus_armed_forces|C)', 'P(nhpi_total|C)', 'P(multi_total|C)']
    Using raw features for Cluster 11 - exactly 2 features: ['P(male_age_low_edu_low|C)', 'P(female_age_low_edu_low|C)']
    Using raw features for Cluster 9 - exactly 2 features: ['P(hispanic|C)', 'P(other_total|C)']
    Skipping Cluster 1 - only 1 feature: P(marital_married|C)
    
    Processing 13 clusters (skipped 1 single-feature clusters)



```python
# Plot scatterplots using top 2 PC's of each cluster, colored by log odds dem repub

# Define colormap
cmap = plt.cm.RdBu
norm = mcolors.Normalize(vmin=df['log_odds_dem_rep'].min(), vmax=df['log_odds_dem_rep'].max())
log_odds = df['log_odds_dem_rep'].values

# Create subplots
n_clusters = len(cluster_features)
fig, axes = plt.subplots(n_clusters, 2, figsize=(15, 5 * n_clusters))

if n_clusters == 1:
    axes = axes.reshape(1, -1)

for i, (cluster_id, features) in enumerate(cluster_features.items()):
    if cluster_methods[cluster_id] == 'raw':
        # Use raw features (exactly 2 features)
        X_unw = scaled_df_unw[features].values
        X_wtd = scaled_df_wtd[features].values
        
        # Plot Unweighted (left)
        ax = axes[i, 0]
        sc = ax.scatter(X_unw[:, 0], X_unw[:, 1], c=log_odds, cmap=cmap, norm=norm, alpha=0.6, s=20)
        ax.set_xlabel(f'{features[0]}')
        ax.set_ylabel(f'{features[1]}')
        ax.set_title(f'Cluster {cluster_id} (Raw Features)')
        ax.grid(True, alpha=0.3)
        
        # Plot Weighted (right)
        ax = axes[i, 1]
        sc = ax.scatter(X_wtd[:, 0], X_wtd[:, 1], c=log_odds, cmap=cmap, norm=norm, alpha=0.6, s=20)
        ax.set_xlabel(f'{features[0]}')
        ax.set_ylabel(f'{features[1]}')
        ax.set_title(f'Cluster {cluster_id} (Weighted Scaled)')
        ax.grid(True, alpha=0.3)
        
    else:  # PCA method
        # Transform data using computed principal components
        X_cluster_unw = scaled_df_unw[features].values
        X_cluster_wtd = scaled_df_wtd[features].values
        
        X_pca_unweighted = X_cluster_unw @ cluster_pcs[cluster_id]['unweighted'].T
        X_pca_weighted = X_cluster_wtd @ cluster_pcs[cluster_id]['weighted'].T
        
        # Plot Unweighted (left)
        ax = axes[i, 0]
        sc = ax.scatter(X_pca_unweighted[:, 0], X_pca_unweighted[:, 1], 
                       c=log_odds, cmap=cmap, norm=norm, alpha=0.6, s=20)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Cluster {cluster_id} (Unweighted PCA)')
        ax.grid(True, alpha=0.3)
        
        # Plot Weighted (right)
        ax = axes[i, 1]
        sc = ax.scatter(X_pca_weighted[:, 0], X_pca_weighted[:, 1], 
                       c=log_odds, cmap=cmap, norm=norm, alpha=0.6, s=20)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Cluster {cluster_id} (Weighted PCA)')
        ax.grid(True, alpha=0.3)

# Add colorbar
cbar = fig.colorbar(sc, ax=axes, orientation='horizontal', fraction=0.02, pad=0.04)
cbar.set_label('Log Odds Democrat vs Republican')

plt.suptitle('Scatterplots of Cluster Components: Unweighted (left) vs Weighted (right)', 
             fontsize=18, fontweight='bold', y=0.95)

# Add cluster feature information as text
cluster_info = []
for cluster_id, features in cluster_features.items():
    method = cluster_methods[cluster_id]
    cluster_info.append(f"Cluster {cluster_id} ({method}): {', '.join(features)}")

print("\nCluster Information:")
for info in cluster_info:
    print(info)

plt.show()

# Save the figure
fig.savefig(PLOT_DIR / 'cluster_pca_scatter_unweighted_weighted.png', bbox_inches='tight', dpi=300)
```

    
    Cluster Information:
    Cluster 7 (pca): median_household_income, per_capita_income, P(income_25k_plus|house_in_C), P(labor_16_plus_employed|C), P(male_age_mid_edu_high|C), P(female_age_mid_edu_high|C), P(male_age_mid_edu_very_high|C), P(female_age_mid_edu_very_high|C)
    Cluster 10 (pca): P(C), population_density, P(asian_total|C), P(marital_single|C), racial_diversity
    Cluster 5 (pca): P(income_10k_to_15k|house_in_C), P(income_15k_to_25k|house_in_C), P(income_less_than_10k|house_in_C), P(poverty|C), P(male_age_mid_edu_low|C), P(female_age_mid_edu_low|C)
    Cluster 8 (pca): P(male_age_low_edu_high|C), P(female_age_low_edu_high|C), P(male_age_low_edu_very_high|C), P(female_age_low_edu_very_high|C)
    Cluster 3 (pca): P(male_age_mid_edu_mid|C), P(female_age_mid_edu_mid|C), P(male_age_high_edu_mid|C), P(female_age_high_edu_mid|C), P(marital_divorced|C)
    Cluster 14 (pca): P(area_in_C), P(aian_total|C), P(male_age_low_edu_mid|C), P(female_age_low_edu_mid|C)
    Cluster 4 (pca): P(labor_16_plus_not_in_force|C), P(male_age_high_edu_low|C), P(female_age_high_edu_low|C), P(marital_widowed|C)
    Cluster 6 (pca): P(male_age_high_edu_high|C), P(female_age_high_edu_high|C), P(male_age_high_edu_very_high|C), P(female_age_high_edu_very_high|C)
    Cluster 2 (raw): P(nativity_native|C), P(white_total|C)
    Cluster 12 (pca): P(labor_16_plus_unemployed|C), P(black_total|C), P(marital_separated|C)
    Cluster 13 (pca): P(labor_16_plus_armed_forces|C), P(nhpi_total|C), P(multi_total|C)
    Cluster 11 (raw): P(male_age_low_edu_low|C), P(female_age_low_edu_low|C)
    Cluster 9 (raw): P(hispanic|C), P(other_total|C)



    
![png](exploratory-data-analysis_files/exploratory-data-analysis_14_1.png)
    



```python
# Plot scatterplots using top PC for each pair of clusters, colored by log odds dem repub

import itertools

# Get all pairs of top 8 clusters
cluster_ids = list(cluster_features.keys())[:8]
pairs = list(itertools.combinations(cluster_ids, 2))

# Create subplots
fig, axes = plt.subplots(len(pairs), 2, figsize=(15, 5 * len(pairs)))

log_odds = df['log_odds_dem_rep'].values
cmap = plt.cm.RdBu
norm = mcolors.Normalize(vmin=df['log_odds_dem_rep'].min(), vmax=df['log_odds_dem_rep'].max())

for i, (cluster_A, cluster_B) in enumerate(pairs):
    features_A = cluster_features[cluster_A]
    features_B = cluster_features[cluster_B]
    
    # Get PC1 or first feature for cluster A
    if cluster_methods[cluster_A] == 'raw':
        pc1_A_unw = scaled_df_unw[features_A[0]].values  # Use first feature
        pc1_A_wtd = scaled_df_wtd[features_A[0]].values
        label_A = features_A[0]
    else:
        X_A_unw = scaled_df_unw[features_A].values
        X_A_wtd = scaled_df_wtd[features_A].values
        pc1_A_unw = X_A_unw @ cluster_pcs[cluster_A]['unweighted'][0, :]  # First component
        pc1_A_wtd = X_A_wtd @ cluster_pcs[cluster_A]['weighted'][0, :]
        label_A = f'PC1 Cluster {cluster_A}'
    
    # Get PC1 or first feature for cluster B
    if cluster_methods[cluster_B] == 'raw':
        pc1_B_unw = scaled_df_unw[features_B[0]].values  # Use first feature
        pc1_B_wtd = scaled_df_wtd[features_B[0]].values
        label_B = features_B[0]
    else:
        X_B_unw = scaled_df_unw[features_B].values
        X_B_wtd = scaled_df_wtd[features_B].values
        pc1_B_unw = X_B_unw @ cluster_pcs[cluster_B]['unweighted'][0, :]  # First component
        pc1_B_wtd = X_B_wtd @ cluster_pcs[cluster_B]['weighted'][0, :]
        label_B = f'PC1 Cluster {cluster_B}'
    
    # Plot Unweighted (left)
    ax = axes[i, 0]
    sc = ax.scatter(pc1_A_unw, pc1_B_unw, c=log_odds, cmap=cmap, norm=norm, alpha=0.6, s=20)
    ax.set_xlabel(label_A)
    ax.set_ylabel(label_B)
    ax.set_title(f'Clusters {cluster_A} vs {cluster_B} (Unweighted)')
    ax.grid(True, alpha=0.15)
    
    # Plot Weighted (right)
    ax = axes[i, 1]
    sc = ax.scatter(pc1_A_wtd, pc1_B_wtd, c=log_odds, cmap=cmap, norm=norm, alpha=0.6, s=20)
    ax.set_xlabel(label_A)
    ax.set_ylabel(label_B)
    ax.set_title(f'Clusters {cluster_A} vs {cluster_B} (Weighted)')
    ax.grid(True, alpha=0.15)

# Add colorbar
cbar = fig.colorbar(sc, ax=axes, orientation='horizontal', fraction=0.02, pad=0.04)
cbar.set_label('Log Odds Democrat vs Republican')

plt.suptitle('PC1 Scatterplots for Cluster Pairings: Unweighted (left) vs Weighted (right)', 
             fontsize=18, fontweight='bold', y=0.9)

# Print cluster pair information
print("\nCluster Pair Information:")
for cluster_A, cluster_B in pairs:
    features_A = cluster_features[cluster_A]
    features_B = cluster_features[cluster_B]
    method_A = cluster_methods[cluster_A]
    method_B = cluster_methods[cluster_B]
    print(f"Pair: Cluster {cluster_A} ({method_A}) vs Cluster {cluster_B} ({method_B})")
    print(f"  Cluster {cluster_A}: {', '.join(features_A)}")
    print(f"  Cluster {cluster_B}: {', '.join(features_B)}")
    print()

plt.show()

# Save the figure
fig.savefig(PLOT_DIR / 'cluster_pairs_pc1_scatter_unweighted_weighted.png', bbox_inches='tight', dpi=300)
```

    
    Cluster Pair Information:
    Pair: Cluster 7 (pca) vs Cluster 10 (pca)
      Cluster 7: median_household_income, per_capita_income, P(income_25k_plus|house_in_C), P(labor_16_plus_employed|C), P(male_age_mid_edu_high|C), P(female_age_mid_edu_high|C), P(male_age_mid_edu_very_high|C), P(female_age_mid_edu_very_high|C)
      Cluster 10: P(C), population_density, P(asian_total|C), P(marital_single|C), racial_diversity
    
    Pair: Cluster 7 (pca) vs Cluster 5 (pca)
      Cluster 7: median_household_income, per_capita_income, P(income_25k_plus|house_in_C), P(labor_16_plus_employed|C), P(male_age_mid_edu_high|C), P(female_age_mid_edu_high|C), P(male_age_mid_edu_very_high|C), P(female_age_mid_edu_very_high|C)
      Cluster 5: P(income_10k_to_15k|house_in_C), P(income_15k_to_25k|house_in_C), P(income_less_than_10k|house_in_C), P(poverty|C), P(male_age_mid_edu_low|C), P(female_age_mid_edu_low|C)
    
    Pair: Cluster 7 (pca) vs Cluster 8 (pca)
      Cluster 7: median_household_income, per_capita_income, P(income_25k_plus|house_in_C), P(labor_16_plus_employed|C), P(male_age_mid_edu_high|C), P(female_age_mid_edu_high|C), P(male_age_mid_edu_very_high|C), P(female_age_mid_edu_very_high|C)
      Cluster 8: P(male_age_low_edu_high|C), P(female_age_low_edu_high|C), P(male_age_low_edu_very_high|C), P(female_age_low_edu_very_high|C)
    
    Pair: Cluster 7 (pca) vs Cluster 3 (pca)
      Cluster 7: median_household_income, per_capita_income, P(income_25k_plus|house_in_C), P(labor_16_plus_employed|C), P(male_age_mid_edu_high|C), P(female_age_mid_edu_high|C), P(male_age_mid_edu_very_high|C), P(female_age_mid_edu_very_high|C)
      Cluster 3: P(male_age_mid_edu_mid|C), P(female_age_mid_edu_mid|C), P(male_age_high_edu_mid|C), P(female_age_high_edu_mid|C), P(marital_divorced|C)
    
    Pair: Cluster 7 (pca) vs Cluster 14 (pca)
      Cluster 7: median_household_income, per_capita_income, P(income_25k_plus|house_in_C), P(labor_16_plus_employed|C), P(male_age_mid_edu_high|C), P(female_age_mid_edu_high|C), P(male_age_mid_edu_very_high|C), P(female_age_mid_edu_very_high|C)
      Cluster 14: P(area_in_C), P(aian_total|C), P(male_age_low_edu_mid|C), P(female_age_low_edu_mid|C)
    
    Pair: Cluster 7 (pca) vs Cluster 4 (pca)
      Cluster 7: median_household_income, per_capita_income, P(income_25k_plus|house_in_C), P(labor_16_plus_employed|C), P(male_age_mid_edu_high|C), P(female_age_mid_edu_high|C), P(male_age_mid_edu_very_high|C), P(female_age_mid_edu_very_high|C)
      Cluster 4: P(labor_16_plus_not_in_force|C), P(male_age_high_edu_low|C), P(female_age_high_edu_low|C), P(marital_widowed|C)
    
    Pair: Cluster 7 (pca) vs Cluster 6 (pca)
      Cluster 7: median_household_income, per_capita_income, P(income_25k_plus|house_in_C), P(labor_16_plus_employed|C), P(male_age_mid_edu_high|C), P(female_age_mid_edu_high|C), P(male_age_mid_edu_very_high|C), P(female_age_mid_edu_very_high|C)
      Cluster 6: P(male_age_high_edu_high|C), P(female_age_high_edu_high|C), P(male_age_high_edu_very_high|C), P(female_age_high_edu_very_high|C)
    
    Pair: Cluster 10 (pca) vs Cluster 5 (pca)
      Cluster 10: P(C), population_density, P(asian_total|C), P(marital_single|C), racial_diversity
      Cluster 5: P(income_10k_to_15k|house_in_C), P(income_15k_to_25k|house_in_C), P(income_less_than_10k|house_in_C), P(poverty|C), P(male_age_mid_edu_low|C), P(female_age_mid_edu_low|C)
    
    Pair: Cluster 10 (pca) vs Cluster 8 (pca)
      Cluster 10: P(C), population_density, P(asian_total|C), P(marital_single|C), racial_diversity
      Cluster 8: P(male_age_low_edu_high|C), P(female_age_low_edu_high|C), P(male_age_low_edu_very_high|C), P(female_age_low_edu_very_high|C)
    
    Pair: Cluster 10 (pca) vs Cluster 3 (pca)
      Cluster 10: P(C), population_density, P(asian_total|C), P(marital_single|C), racial_diversity
      Cluster 3: P(male_age_mid_edu_mid|C), P(female_age_mid_edu_mid|C), P(male_age_high_edu_mid|C), P(female_age_high_edu_mid|C), P(marital_divorced|C)
    
    Pair: Cluster 10 (pca) vs Cluster 14 (pca)
      Cluster 10: P(C), population_density, P(asian_total|C), P(marital_single|C), racial_diversity
      Cluster 14: P(area_in_C), P(aian_total|C), P(male_age_low_edu_mid|C), P(female_age_low_edu_mid|C)
    
    Pair: Cluster 10 (pca) vs Cluster 4 (pca)
      Cluster 10: P(C), population_density, P(asian_total|C), P(marital_single|C), racial_diversity
      Cluster 4: P(labor_16_plus_not_in_force|C), P(male_age_high_edu_low|C), P(female_age_high_edu_low|C), P(marital_widowed|C)
    
    Pair: Cluster 10 (pca) vs Cluster 6 (pca)
      Cluster 10: P(C), population_density, P(asian_total|C), P(marital_single|C), racial_diversity
      Cluster 6: P(male_age_high_edu_high|C), P(female_age_high_edu_high|C), P(male_age_high_edu_very_high|C), P(female_age_high_edu_very_high|C)
    
    Pair: Cluster 5 (pca) vs Cluster 8 (pca)
      Cluster 5: P(income_10k_to_15k|house_in_C), P(income_15k_to_25k|house_in_C), P(income_less_than_10k|house_in_C), P(poverty|C), P(male_age_mid_edu_low|C), P(female_age_mid_edu_low|C)
      Cluster 8: P(male_age_low_edu_high|C), P(female_age_low_edu_high|C), P(male_age_low_edu_very_high|C), P(female_age_low_edu_very_high|C)
    
    Pair: Cluster 5 (pca) vs Cluster 3 (pca)
      Cluster 5: P(income_10k_to_15k|house_in_C), P(income_15k_to_25k|house_in_C), P(income_less_than_10k|house_in_C), P(poverty|C), P(male_age_mid_edu_low|C), P(female_age_mid_edu_low|C)
      Cluster 3: P(male_age_mid_edu_mid|C), P(female_age_mid_edu_mid|C), P(male_age_high_edu_mid|C), P(female_age_high_edu_mid|C), P(marital_divorced|C)
    
    Pair: Cluster 5 (pca) vs Cluster 14 (pca)
      Cluster 5: P(income_10k_to_15k|house_in_C), P(income_15k_to_25k|house_in_C), P(income_less_than_10k|house_in_C), P(poverty|C), P(male_age_mid_edu_low|C), P(female_age_mid_edu_low|C)
      Cluster 14: P(area_in_C), P(aian_total|C), P(male_age_low_edu_mid|C), P(female_age_low_edu_mid|C)
    
    Pair: Cluster 5 (pca) vs Cluster 4 (pca)
      Cluster 5: P(income_10k_to_15k|house_in_C), P(income_15k_to_25k|house_in_C), P(income_less_than_10k|house_in_C), P(poverty|C), P(male_age_mid_edu_low|C), P(female_age_mid_edu_low|C)
      Cluster 4: P(labor_16_plus_not_in_force|C), P(male_age_high_edu_low|C), P(female_age_high_edu_low|C), P(marital_widowed|C)
    
    Pair: Cluster 5 (pca) vs Cluster 6 (pca)
      Cluster 5: P(income_10k_to_15k|house_in_C), P(income_15k_to_25k|house_in_C), P(income_less_than_10k|house_in_C), P(poverty|C), P(male_age_mid_edu_low|C), P(female_age_mid_edu_low|C)
      Cluster 6: P(male_age_high_edu_high|C), P(female_age_high_edu_high|C), P(male_age_high_edu_very_high|C), P(female_age_high_edu_very_high|C)
    
    Pair: Cluster 8 (pca) vs Cluster 3 (pca)
      Cluster 8: P(male_age_low_edu_high|C), P(female_age_low_edu_high|C), P(male_age_low_edu_very_high|C), P(female_age_low_edu_very_high|C)
      Cluster 3: P(male_age_mid_edu_mid|C), P(female_age_mid_edu_mid|C), P(male_age_high_edu_mid|C), P(female_age_high_edu_mid|C), P(marital_divorced|C)
    
    Pair: Cluster 8 (pca) vs Cluster 14 (pca)
      Cluster 8: P(male_age_low_edu_high|C), P(female_age_low_edu_high|C), P(male_age_low_edu_very_high|C), P(female_age_low_edu_very_high|C)
      Cluster 14: P(area_in_C), P(aian_total|C), P(male_age_low_edu_mid|C), P(female_age_low_edu_mid|C)
    
    Pair: Cluster 8 (pca) vs Cluster 4 (pca)
      Cluster 8: P(male_age_low_edu_high|C), P(female_age_low_edu_high|C), P(male_age_low_edu_very_high|C), P(female_age_low_edu_very_high|C)
      Cluster 4: P(labor_16_plus_not_in_force|C), P(male_age_high_edu_low|C), P(female_age_high_edu_low|C), P(marital_widowed|C)
    
    Pair: Cluster 8 (pca) vs Cluster 6 (pca)
      Cluster 8: P(male_age_low_edu_high|C), P(female_age_low_edu_high|C), P(male_age_low_edu_very_high|C), P(female_age_low_edu_very_high|C)
      Cluster 6: P(male_age_high_edu_high|C), P(female_age_high_edu_high|C), P(male_age_high_edu_very_high|C), P(female_age_high_edu_very_high|C)
    
    Pair: Cluster 3 (pca) vs Cluster 14 (pca)
      Cluster 3: P(male_age_mid_edu_mid|C), P(female_age_mid_edu_mid|C), P(male_age_high_edu_mid|C), P(female_age_high_edu_mid|C), P(marital_divorced|C)
      Cluster 14: P(area_in_C), P(aian_total|C), P(male_age_low_edu_mid|C), P(female_age_low_edu_mid|C)
    
    Pair: Cluster 3 (pca) vs Cluster 4 (pca)
      Cluster 3: P(male_age_mid_edu_mid|C), P(female_age_mid_edu_mid|C), P(male_age_high_edu_mid|C), P(female_age_high_edu_mid|C), P(marital_divorced|C)
      Cluster 4: P(labor_16_plus_not_in_force|C), P(male_age_high_edu_low|C), P(female_age_high_edu_low|C), P(marital_widowed|C)
    
    Pair: Cluster 3 (pca) vs Cluster 6 (pca)
      Cluster 3: P(male_age_mid_edu_mid|C), P(female_age_mid_edu_mid|C), P(male_age_high_edu_mid|C), P(female_age_high_edu_mid|C), P(marital_divorced|C)
      Cluster 6: P(male_age_high_edu_high|C), P(female_age_high_edu_high|C), P(male_age_high_edu_very_high|C), P(female_age_high_edu_very_high|C)
    
    Pair: Cluster 14 (pca) vs Cluster 4 (pca)
      Cluster 14: P(area_in_C), P(aian_total|C), P(male_age_low_edu_mid|C), P(female_age_low_edu_mid|C)
      Cluster 4: P(labor_16_plus_not_in_force|C), P(male_age_high_edu_low|C), P(female_age_high_edu_low|C), P(marital_widowed|C)
    
    Pair: Cluster 14 (pca) vs Cluster 6 (pca)
      Cluster 14: P(area_in_C), P(aian_total|C), P(male_age_low_edu_mid|C), P(female_age_low_edu_mid|C)
      Cluster 6: P(male_age_high_edu_high|C), P(female_age_high_edu_high|C), P(male_age_high_edu_very_high|C), P(female_age_high_edu_very_high|C)
    
    Pair: Cluster 4 (pca) vs Cluster 6 (pca)
      Cluster 4: P(labor_16_plus_not_in_force|C), P(male_age_high_edu_low|C), P(female_age_high_edu_low|C), P(marital_widowed|C)
      Cluster 6: P(male_age_high_edu_high|C), P(female_age_high_edu_high|C), P(male_age_high_edu_very_high|C), P(female_age_high_edu_very_high|C)
    



    
![png](exploratory-data-analysis_files/exploratory-data-analysis_15_1.png)
    



```python

```
