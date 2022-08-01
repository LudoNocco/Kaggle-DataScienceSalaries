from cProfile import run
from typing import List
from itertools import combinations
import os

import numpy as np
import pandas as pd
 
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

from ipdb import set_trace

def run_analysis(dataframe: pd.DataFrame, prefix: str):

    variables = dataframe.columns
    pairs = combinations(variables, 2)
    pairs = [pair for pair in pairs]
    
    df_stats = pd.DataFrame(columns=["pair", "spearmann", "p_value"])
    for ind, pair in enumerate(pairs):
        variable1, variable2 = pair 

        correlation = spearmanr(dataframe[variable1], dataframe[variable2])
        corr, pvalue = correlation

        df_stats.at[ind, "variable_1"] = variable1 
        df_stats.at[ind, "variable_2"] = variable2
        df_stats.at[ind, "spearmann"] = corr
        df_stats.at[ind, "p_value"] = pvalue

        df_significant = df_stats[df_stats["p_value"] <= 0.01]

        df_significant.to_csv(f"results/significant_correlations_{prefix}.csv", index=False)

    #training_features = ["experience_level", "employment_type", "job_title"]
    training_features_1 = list(df_significant["variable_1"].values)
    training_features_2 = list(df_significant["variable_2"].values)
    training_features = training_features_1 + training_features_2
    training_features = list(np.unique(training_features))

    target_variable = "salary_in_usd" 

    if target_variable in training_features:
        training_features.remove(target_variable)
    for feature in training_features:
        if "salary" in feature:
            training_features.remove(feature)

    run_linear_regression(data=df, training_variables=training_features, target_variable=target_variable)

def run_linear_regression(data: pd.DataFrame, training_variables: List, target_variable: str):

    # Shuffle data
    data = data.sample(frac = 1)

    # Extract target and features
    target = np.log(data[target_variable])
    variables_num = []
    variables_cat = []
    for var in training_variables:
        if isinstance(data.loc[0, var], str):
            variables_cat.append(var)
        else:
            variables_num.append(var)

    print("Numerical")
    print(variables_num)
    print("Categorical")
    print(variables_cat)

    features_num = data[variables_num]
    features_cat = data[variables_cat]
    
    # Preprocess training data
    features_cat = pd.get_dummies(features_cat)
    features = pd.concat([features_num, features_cat], axis=1)

    # Get amount of trainning and test instances
    tot_train = int(features.shape[0] * 0.8)
    tot_test = int(features.shape[0] * 0.2)

    # Sample training and test data
    X_train = features.head(tot_train)
    X_test = features.tail(tot_test)
     
    Y_train = target.head(tot_train)
    Y_test = target.tail(tot_test)
    
    # Model training (regression)
    model = LinearRegression()
    model.fit(X_train.values, Y_train.values)

    # Model evluation
    predictions = model.predict(X_test.values)
    mse = mean_squared_error(Y_test.values, predictions)
    print("Linear Regression")
    print(f"Mean Squared Error: {mse}")

    # Model training (randomforest)
    model = RandomForestRegressor()
    model.fit(X_train.values, Y_train.values)

    # Model evluation
    predictions = model.predict(X_test.values)
    mse = mean_squared_error(Y_test.values, predictions)
    print("Random Forest Regressor")
    print(f"Mean Squared Error: {mse}")

if __name__=="__main__":

    data_path = "data/dsjs.csv"
    assert os.path.isfile(data_path), f"Invalid path for data: {data_path}"

    df = pd.read_csv(data_path)
    
    df["is_estimated"] = df.apply(lambda x: "e" in x["work_year"], axis=1)
    df_measured = df[df["is_estimated"] == False]
    df_estimated = df[df["is_estimated"] == True]

    df = df.drop(columns=["is_estimated"])
    df_measured = df_measured.drop(columns=["is_estimated", "work_year"])
    df_estimated = df_estimated.drop(columns=["is_estimated", "work_year"])

    dataframes = [df, df_measured, df_estimated]
    prefixes = ["all", "measured", "estimated"]
    for prefix, dataframe in zip(prefixes, dataframes):
        run_analysis(dataframe, prefix)        

