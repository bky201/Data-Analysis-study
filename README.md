# Data Analysis

This project, includes the implementation of various data analysis methods and compare the result of each implementation for sales prediction and visualization. Data is analized using different approaches inorder to determine which approach works the best
with our data analysis. The main scope is to identify which algorithms give the best result and which algorithms result in poor data prediction. Determining sales units forecasting using feature scaling and dimensionality reduction methods. Data can be affected by many factors that result in irrelevant and redundant data. This noisy or unreliable data can lead to the wrong prediction if it is not handled correctly. Thus the quality of data is the first and foremost stage in data analysis. To obtain exploitable data, the data must be preprocessed before applied to any machine learning algorithm. Preprocessing of data includes data cleaning, data normalization, data transformation, feature extraction, dimensionality reduction, etc. The result of data processing is the final required data for the training algorithm. The final result indicates that no best method can be implemented on all types of datasets. The best sequence for the data preprocessing algorithm depends on the kind of data and its distribution.

## Table of Contents
* [Data Analysis](#data-analysis)
* [Data Description](#data-description)
* [Data Preprocessing](#data-preprocessing) 
  * [Data analysis software tools](#data-analysis-software-tools)
  * [Handling Missing Values using Multiple Imputation](#handling-missing-values-using-multiple-imputation)
  * [Data Exploration and Visualization](#data-exploration-and-visualization) 
* [Numerical Feature Scaling](#numerical-feature-scaling)
  * [Min-Max Normalization](#min-max-normalization)
  * [Z-Score Normalization](#z-score-normalization)
  * [Robustscaler Normalization](#robustscaler-normalization) 
* [Dimensionality reduction](#dimensionality-reduction)
  * [Autoencoder](#autoencoder) 
  * [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
  * [Truncated Singular Value Decomposition (SVD)](#truncated-singular-value-decomposition-svd)
  * [Gaussian and Sparse Random Projection](#gaussian-and-sparse-random-projection) 
* [Regression Models](#regression-models)
  * [Extreme Gradient Boosting (XGBoost)](#extreme-gradient-boosting-xgboost)
  * [Multilayer Perceptron (MLP)](#multilayer-perceptron-mlp)
  * [Random Forest](#random-forest)
* [Model Evaluation Metric](#model-evaluation-metric)
  * [R-squared](#r-squared)
  * [Root Mean Square Error (RMSE)](#root-mean-square-error-rmse)
* [Data Cleaning](#data-cleaning)
  * [Data Type Modification](#data-type-modification)
* [Data Distribution and Exploration](#data-distribution-and-exploration)
  * [Data Distribution with Min-Max Normalization](#data-distribution-with-min-max-normalization)
  * [Data Distribution with Z-Score Normalization](#data-distribution-with-z-score-normalization)
  * [Data Distribution with Robustscaler Normalization](#data-distribution-with-robustscaler-normalization)
* [Correlation between features](#correlation-between-features)
* [Feature Scaling and Dimensionality Reduction](#feature-scaling-and-dimensionality-reduction)
  * [Feature Scaling followed by Autoencoder Dimensionality Reudction](#feature-scaling-followed-by-autoencoder-dimensionality-reudction)
  * [Feature Scaling followed by Principal Component Analysis (PCA)](#feature-scaling-followed-by-principal-component-analysis-pca)
  * [Feature Scaling followed by Truncated Singular Value Decomposition (Truncated SVD)](#feature-scaling-followed-by-truncated-singular-value-decomposition-truncated-svd)
  * [Feature Scaling followed by Gaussian Random Projection](#feature-scaling-followed-by-gaussian-random-projection)
  * [Feature Scaling followed by Sparse Random Projection](#feature-scaling-followed-by-sparse-random-projection)
* [Results and Discussion](#results-and-discussion)




## Data Description 

The data used in this study is available online from the [kaggle](https://www.kaggle.com/datasets/dgluesen/sales-and-workload-data-from-retail-industry) website. The raw data contains generalized information about a real-world retail stores which is often used for actual analytical processing. Most industries’ data format is usually not
suitable to be processed directly in machine learning models. As a result, such data must be preprocessed in such a way that the algorithm can easily understand and interpret the features of data. Another important aspect of data preprocessing is
that it reduces the data by obtaining the most relevant features, thus making the learning algorithm to consume less memory and process faster.

#### Overview of the top 5 entries of the dataset 

![fig-31](./result-images/fig-31.png)

## Data Preprocessing 

#### Architecture of data preprocessing flow chart 

<p align="center"><img src="./result-images/fig-32.png"
        alt="flow chart"></p>

### Data analysis software tools 

Data analysis was completed using Python 3.6 in Jupyter notebook along with the following modules:

• Pandas for Data analysis and manipulation

• NumPy for Numerical computation

• Matplotlib, Seaborn for Visualisation and plotting

• Scikit-Learn, Keras for Machine learning

### Handling Missing Values using Multiple Imputation 

Missing data can be handled in a variety of ways. Parameter estimation, single value imputation and multiple imputation (MI) techniques are the most common techniques used for handling missing values. MI is one of the best methods for handling missing values.

![algo-1](./result-images/algo-1.png)

### Data Exploration and Visualization 

When we are working on a data analysis project and want to get a quick first-hand impression of our data, we can achieve that in two key steps: Data Exploration and Data Visualization. Data Exploration is an essential part of Exploratory Data Analysis (EDA). Depending on the general information we observe at the data exploration step, we can decide what to do in the subsequent steps. Therefore, to select our machine learning model and summarize our findings, the assumptions we make at the exploration phase plays an essential role.

Some of the primary and widely used charts are: 

 - Bar graph 
 - Line graph
 - Heatmap  
 - Histogram 
 - Scatter plot, and 
 - Box plot.

## Numerical Feature Scaling 

Feature scaling is a technique for narrowing the range of continuous variables to a defined interval, such as [0,1] or [1,-1]. It is a typical data transformation technique used in data preprocessing and one of the classification methods that require distance measurements such as PCA, KNN, and SVM models. Furthermore, feature scaling accelerates the NN backpropagation technique for classification problems.

### Min-Max Normalization 

Min-Max scaling is a method for limiting the size of a feature to a specified range of [-1, 1] or [0, 1].

### Z-Score Normalization 

The Z-score is the most commonly used approach for converting all variables to a standard scale by subtracting the mean value and scale it with the standard deviation.

### Robustscaler Normalization 

This technique is the same as Min-max normalization, except it scales the data variables using interquartile values rather than min and max values.

## Dimensionality reduction 

Dimensionality reduction is a preprocessing stage in which high-dimensional data is converted into lower-dimensional data. Our analysis method utilizes five different algorithms: autoencoders, principal component analysis (PCA), Gaussian random projection, Sparse random projection, and Truncated Singular Value decomposition (Truncated SVD).

### Autoencoder 

![algo-2](./result-images/algo-2.png)

![algo-4](./result-images/algo-4.png)

#### Autoencoder architecture 

![fig-33](./result-images/fig-33.png)

### Principal Component Analysis (PCA) 

![algo-3](./result-images/algo-3.png)

### Truncated Singular Value Decomposition (SVD) 

![algo-5](./result-images/algo-5.png)

#### Truncated Singular Values Decomposition 

![fig-34](./result-images/fig-34.png)

### Gaussian and Sparse Random Projection 

![algo-6](./result-images/algo-6.png)


## Regression Models 
A regression problem is a technique that uses an independent variable to estimate the quantitative output of a dependent variable. The type of regression models used in this section is nonparametric regressions models such as eXtreme Gradient Boosting (XGBoost), Random Forest and deep neural networks (e.g., Multilayer Perceptrons (MLP)).

### Extreme Gradient Boosting (XGBoost) 

The XGBoost method is a more advanced version of the gradient boosting technique. It is an advanced machine learning algorithm for classification and regression problems that uses an end-to-end tree boosting strategy.

### Multilayer Perceptron (MLP) 

Multilayer perceptrons (MLPs) are deep feed-forward neural networks specifically developed for multi-class supervised learning tasks. MLP network is trained on a collection of input-output pairs to learn how to represent the relationships between those inputs and outputs.

### Random Forest 

It is a machine learning ensemble algorithm that may be used to create a classification or regression model. RF fits multiple trees by picking a random subset (bootstrap sample) of the predictors from the original data and provides a classification or regression prediction. Finally, the output is defined by the initial issue description, the mode of the classes for classification or the mean forecast for regression.

## Model Evaluation Metric 
### R-squared 

The coefficient of determination in statistics is the fraction of the variation in the dependent variable that can be predicted by the independent variable. In the best-case scenario, the modeled values precisely match the observed values, resulting in squared sum error of regression (Sr) = 0 and R-squared = 1. A baseline model, which always predicts the targets, will have R-squared = 0. A negative R2 indicates a predictions that are worse than the baseline.

### Root Mean Square Error (RMSE)

The root-mean-square error (RMSE) is a commonly used metric for comparing values predicted by a model or estimator to values observed. The square root of the mean values of the square of all errors gives the root mean squared error (RMSE).

## Data Cleaning 

Most of the data collected from different warehouses are potentially not clean and contain incomplete data. The first step to take after uploading the data set is to clean the data because not cleaned data can lead to a range of problems, such as model miss prediction, errors in parameter estimation and wrong conclusions about the data analysis result.

### Data Type Modification 

The data type should indicate the correct format of the values in each column. For example, numeric columns can be assigned a strings or vice versa which is a wrong identification. Additionally, categorical data should convert to ’category’ data type, making the dataset smaller in memory and more uncomplicated to utilize in further analysis.

#### Data type before cleaning 

![fig-41](./result-images/fig-41.png)

#### Percentage of missing values 

![fig-42](./result-images/fig-42.png)

#### Nullity Matrix 

The nullity matrix helps represent the data in dense layers with a sparkline at the right end that gives the general information of the data by removing rows with missing data out of the row.

![fig-43](./result-images/fig-43.png)

#### Heat map 

Heatmap is a method of visualizing the correlation between missing values. If the value of correlation between two
variables is 1, it means both variables contain missing values. If the value of correlation is 0, it means that a variable that exists or does not exist does not
depend on each other.

![fig-44](./result-images/fig-44.png)

#### Dendrogram of missing data 

It is a hierarchical representation between features. A dendrogram represents the data in hierarchical clustering. All the variables except the customer variable belong to one cluster. The similarity between variables increases if the total distance between the variable or the average distance along the y-axis is much smaller than zero. The dendrogram arranges similar features near each other in the tree.

![fig-45](./result-images/fig-45.png)

## Data Distribution and Exploration 

#### Probability Density of a continuous variable distribution

![fig-46](./result-images/fig-46.png)

#### Bar graph for monthly sales distribution vs. country of sales

![fig-47](./result-images/fig-47.png)

#### Bar graph for monthly sales distribution vs. sales item

![fig-48](./result-images/fig-48.png)

#### Bar graph for monthly sales distribution vs. city of sales

![fig-49](./result-images/fig-49.png)

#### Scatter plot for HoursLease vs. HoursOwn

![fig-410](./result-images/fig-410.png)

#### Scatter pair plot of continuous variables

![fig-411](./result-images/fig-411.png)

#### Boxplots for sales Units vs. Department name

![fig-412](./result-images/fig-412.png)

#### Boxplots for Hours own vs. Department name 

![fig-413](./result-images/fig-413.png)

#### Boxplots for Hours Lease vs. Department name 

![fig-414](./result-images/fig-414.png)

#### Boxplots for Turnover vs. Department name 

![fig-415](./result-images/fig-415.png)

#### Boxplots for Area vs. Department name 

![fig-416](./result-images/fig-416.png)

### Data Distribution with Min-Max Normalization 

#### Data distribution after minmax normalization 

![fig-417](./result-images/fig-417.png)

#### Scatter pair plot after minmax normalization 

![fig-420](./result-images/fig-420.png)

### Data Distribution with Z-Score Normalization 

#### Data distribution after Z-Score Normalization 

![fig-418](./result-images/fig-418.png)

#### Scatter pair plot after Z-score Normalization 

![fig-421](./result-images/fig-421.png)

### Data Distribution with Robustscaler Normalization 

#### Data distribution after Robustscaler Normalization

![fig-419](./result-images/fig-419.png)

#### Scatter pair plot after Robustscaler normalization 

![fig-422](./result-images/fig-422.png)

## Correlation between features 

#### Heatmap of continuous variables 

![fig-423](./result-images/fig-423.png)

## Feature Scaling and Dimensionality Reduction 

### Feature Scaling followed by Autoencoder Dimensionality Reudction 

#### Scatter plot of Autoencoder with Normalization 

![fig-424](./result-images/fig-424.png)

#### Scatter plot of Auroencoder with Robustscaler Normalization 

![fig-425](./result-images/fig-425.png)

#### Scatter plot of Autoencoder with Standardization 

![fig-426](./result-images/fig-426.png)

### Feature Scaling followed by Principal Component Analysis (PCA) 

#### Scater plot of PCA with Normalization 

![fig-427](./result-images/fig-427.png)

#### Scatter plot of PCA with RobustScaler 

![fig-428](./result-images/fig-428.png)

#### Scatter plot of PCA with Standardization 

![fig-429](./result-images/fig-429.png)

### Feature Scaling followed by Truncated Singular Value Decomposition (Truncated SVD) 

#### Scatter plot of Truncated SVD after Normalization 

![fig-430](./result-images/fig-430.png)

#### Scatter plot of Truncated SVD with RobustScaler 

![fig-431](./result-images/fig-431.png)

#### Scatter plot of Truncated SVD with Standardization 

![fig-432](./result-images/fig-432.png)

### Feature Scaling followed by Gaussian Random Projection 

#### Scater plot of Gaussian Random Projection with normalization 

![fig-433](./result-images/fig-433.png)

#### Scatter plot of Gaussian Random Projection with RobustScaler 

![fig-434](./result-images/fig-434.png)

#### Scatter plot of Gaussian Random Projection with Z-score normalization

![fig-435](./result-images/fig-435.png)

### Feature Scaling followed by Sparse Random Projection 

#### Scatter plot of Sparse Random Projection with Normalization 

![fig-436](./result-images/fig-436.png)

#### Scatter plot of Sparse Random Projection with RobustScaler 

![fig-437](./result-images/fig-437.png)

#### Scatter plot of Sparse Random Projection with Standardization

![fig-438](./result-images/fig-438.png)

## Results and Discussion
 
#### R-squared value for minmax normalized data with different dimensionality reduction methods on three models

![fig-51](./result-images/fig-51.png)

#### R-squared value for Robustscaler normalized data with different dimensionality reduction methods on three models

![fig-52](./result-images/fig-52.png)

#### R-squared value for Z score standardized data with different dimensionality reduction methods on three models 

![fig-53](./result-images/fig-53.png)

#### RMSE value for minmax normalized data with different dimensionality reduction methods on three models 

![fig-54](./result-images/fig-54.png)

#### RMSE value for Robustscaler normalized data with different dimensionality reduction methods on three models 

![fig-55](./result-images/fig-55.png)

#### RMSE value for Z score standardized data with different dimensionality reduction methods on three models 

![fig-56](./result-images/fig-56.png)
