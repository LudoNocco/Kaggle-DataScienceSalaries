from typing import List
from itertools import combinations
import os
import numpy as np
import pandas as pd
 
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

def run_analysis(dataframe_path: str):

    assert os.path.isfile(dataframe_path), f"Invalid path for data: {dataframe_path}"

    df = pd.read_csv(data_path)

    variables = df.columns
    pairs = combinations(variables, 2)
    pairs = [pair for pair in pairs]
    
    df_stats = pd.DataFrame(columns=["pair", "spearmann", "p_value"])
    for ind, pair in enumerate(pairs):
        variable1, variable2 = pair 

        correlation = spearmanr(df[variable1], df[variable2])
        corr, pvalue = correlation

        df_stats.at[ind, "pair"] = variable1 + " & " + variable2
        df_stats.at[ind, "spearmann"] = corr
        df_stats.at[ind, "p_value"] = pvalue

        df_significant = df_stats[df_stats["p_value"] <= 0.01]

        df_significant.to_csv("results/significant_correlations.csv", index=False)

    training_features = ["experience_level", "employment_type", "job_title"]
    target_variable = "salary_in_usd" 

    run_linear_regression(data=df, training_features=training_features, target_variable=target_variable)

def run_linear_regression(data: pd.DataFrame, training_features: List, target_variable: str):

    # Shuffle data
    data = data.sample(frac = 1)

    # Extract target and features
    target = data[target_variable]
    features = data[training_features]
    
    # Preprocess training data
    features = pd.get_dummies(features)

    # Get amount of trainning and test instances
    tot_train = int(features.shape[0] * 0.8)
    tot_test = int(features.shape[0] * 0.2)

    # Sample training and test data
    X_train = features.head(tot_train)
    X_test = features.tail(tot_test)
     
    Y_train = target.head(tot_train)
    Y_test = target.tail(tot_test)
    
    # Model training
    model = LinearRegression()
    model.fit(X_train.values, Y_train.values)

    # Model evluation
    predictions = model.predict(X_test.values)
    mse = mean_squared_error(Y_test.values, predictions)
    print(f"Mean Squared Error: {mse}")
    print(predictions)
    print(Y_test)

if __name__=="__main__":
    data_path = "data/dsjs.csv"
    run_analysis(data_path)