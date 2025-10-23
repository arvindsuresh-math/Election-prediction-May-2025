# US County-Level Election Prediction

## Project Overview

This project aims to predict US county-level presidential election outcomes (popular vote shares) using demographic data. The goal is to build accurate models that can forecast voting patterns based on socioeconomic, racial, educational, and other demographic features at the county level.

## Modeling Strategy

The project employs a comparative approach, evaluating multiple machine learning models to identify the best performing algorithm for county-level election prediction:

- **Ridge Regression**: Baseline linear model with L2 regularization, tested with and without PCA dimensionality reduction
- **XGBoost**: Gradient boosting tree-based model with hyperparameter optimization
- **Softmax Regression**: Multi-class classification approach for the four voting categories (Democrat, Republican, Other, Non-voter)
- **Multi-Layer Perceptron (MLP)**: Neural network models with varying depths (1, 2, and 3 hidden layers)

All models use Optuna for automated hyperparameter optimization. The evaluation focuses on weighted metrics that account for county population sizes, ensuring predictions are representative of the actual electorate rather than just geographic units.

## Exploratory Data Analysis Observations

The EDA revealed several key insights about the demographic data and its relationship to voting patterns:

- **Correlation Analysis**: Significant differences exist between population-weighted and unweighted correlations, indicating that weighting by county population is crucial for accurate analysis. Features like county population (P(C)), area, and population density show substantial correlation changes when weighted.

- **Feature Clustering**: Hierarchical clustering based on correlation distances grouped features into meaningful categories including demographic groups (income, ethnicity, race, education, marital status) and labor force characteristics.

- **Principal Component Analysis**: PCA on demographic groups and feature clusters revealed patterns in how different demographic factors relate to voting behavior. The analysis showed clear separations between demographic categories and their associations with political preferences.

## Final Evaluation Results

| Model          | p_dem  | p_rep  | p_other | p_nonvoter | log_odds_dem_vs_rep | weighted_entropy | weighted_kl | weighted_kl_percent | weighted_accuracy |
|----------------|--------|--------|---------|------------|---------------------|------------------|-------------|---------------------|-------------------|
| true values    | 0.2434 | 0.2253 | 0.0090  | 0.5223     | 0.0770              | 1.0206          | 0.0000     | 0.0000             | 1.0000           |
| ridge_base     | 0.2071 | 0.1802 | 0.0299  | 0.5828     | 0.1391              | 1.0232          | 0.0383     | 0.0409             | 0.8952           |
| ridge_pca      | 0.2071 | 0.1802 | 0.0299  | 0.5829     | 0.1391              | 1.0231          | 0.0383     | 0.0409             | 0.8952           |
| xgboost        | 0.2087 | 0.1813 | 0.0292  | 0.5808     | 0.1406              | 1.0184          | 0.0300     | 0.0303             | 0.9252           |
| softmax        | 0.2263 | 0.1735 | 0.0424  | 0.5578     | 0.2659              | 1.0610          | 0.0399     | 0.0411             | 0.9029           |
| mlp_1          | 0.2102 | 0.1701 | 0.0298  | 0.5899     | 0.2117              | 1.0032          | 0.0356     | 0.0361             | 0.9127           |
| mlp_2          | 0.2197 | 0.1651 | 0.0399  | 0.5753     | 0.2860              | 1.0473          | 0.0417     | 0.0419             | 0.8912           |
| mlp_3          | 0.2096 | 0.1952 | 0.0343  | 0.5609     | 0.0714              | 1.0658          | 0.0293     | 0.0297             | 0.9168           |

**Key Metrics Explained:**

- **p_dem/p_rep/p_other/p_nonvoter**: Weighted average predicted probabilities for each voting category
- **log_odds_dem_vs_rep**: Log odds ratio between Democrat and Republican probabilities
- **weighted_entropy**: Average entropy of predicted distributions across counties
- **weighted_kl**: Kullback-Leibler divergence between predicted and true distributions
- **weighted_kl_percent**: KL divergence as percentage of true distribution entropy
- **weighted_accuracy**: Population-weighted fraction of correctly predicted county winners

The XGBoost model achieved the highest weighted accuracy (92.52%) and lowest KL divergence (3.03%), making it the best performing model overall. Yet (perhaps paradoxically), the 3-layer MLP model was able to model the true distribution most closely; indeed, its KL divergence was 2.97% and its predicted log-odds of voting democrat vs republican was 0.0714, compared to the true value of 0.0770 and XGBoost's predicted value of 0.1406!
