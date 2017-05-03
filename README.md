# EuropeanFootballMatchPrediction
Predicting match outcomes of European football games by Rohan Sunder and Manuel Perez.

The main files of interest are located in the root of this repository. Code files in the folders were used more or less as utility code to 
generate the final datasets. The datasets were ingested using R and WEKA with some Neural Net experimentation using Python (3.4.4) Theano 
in the European_Football_Data_Mining folder. After finding no noticeable benefits from the Theano implementation, development was halted.

dim_reduction_rankings.txt - This file provides the rankings of the features from most useful to least.
randomized_final_dataset.csv - This was the first final dataset that was ingested into R and WEKA. The first round of results used this dataset.
randomized_final_dataset_v2.csv - This was the second final dataset that was ingested into R and WEKA. The second round of results used this dataset.
pca_all_data.csv - This file was generated after running PCA on randomized_final_dataset_v2.csv. This file is meant to be used to generate results.
