from pathlib import Path
from scipy.sparse import data
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  GridSearchCV, KFold, train_test_split
from sklearn.metrics import  mean_absolute_error, mean_squared_error, r2_score

import joblib
import json
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tabular_data as td
import time
import xgboost as xgb

pd.options.mode.chained_assignment = None

def return_model_performance_metrics(y_train, y_train_pred, y_validation, y_validation_pred, y_test, y_test_pred):
    train_pred_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2_pred = r2_score(y_train,y_train_pred)
    train_mae_pred = mean_absolute_error(y_train, y_train_pred)
    
    validation_rmse_pred = math.sqrt(mean_squared_error(y_validation, y_validation_pred))
    validation_r2_pred = r2_score(y_validation,y_validation_pred)
    validation_mae_pred = mean_absolute_error(y_validation, y_validation_pred)

    test_rmse_pred = math.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2_pred = r2_score(y_test,y_test_pred)
    test_mae_pred = mean_absolute_error(y_test, y_test_pred)


    performance_metrics_dict = {"train_pred_rmse": train_pred_rmse, "train_r2_pred": train_r2_pred, "train_mae_pred": train_mae_pred,
                                "validation_rmse_pred": validation_rmse_pred, "validation_r2_pred": validation_r2_pred, "validation_mae_pred": validation_mae_pred,
                                "test_rmse_pred": test_rmse_pred, "test_r2_pred": test_r2_pred, "test_mae_pred": test_mae_pred}
    return performance_metrics_dict

def save_model(model, hyperparameters, metrics, folder_for_files):
    
    joblib.dump(model, f'{folder_for_files}/model.joblib')
    with open(f'{folder_for_files}/hyperparameters.json', 'w') as outfile:
        json.dump(hyperparameters, outfile)
    with open(f'{folder_for_files}\metrics.json', 'w') as outfile:
        json.dump(metrics, outfile)

def split_the_data(features, labels, test_to_rest_ratio, validation_to_test_ratio):
    X = features
    y = labels
    #split the data into test and train data; the 0.3 describes the data which is apportioned to the test set 
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_to_rest_ratio, random_state=13)
    #print(y_train)
    #resplit the test data again to get a final 15 % for both test and validation
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=validation_to_test_ratio, random_state=13)
    
    return [X_train, y_train, X_validation, y_validation, X_test, y_test]

def get_baseline_score(regression_model, data_subsets, folder):
    """Tunes the hyperparameters of a regression model and saves the information.
    Args:
        model (class): The regression model to be as a baseline.
        data_subsets (list): List in the form [X_train, y_train, X_validation,
            y_validation, X_test, y_test].
        folder (str): The directory path of where to save the data.
    Returns:
        metrics (dict): Training, validation and testing performance metrics.
    """
    logging.info('Calculating baseline score...')
    #fit the data baes on labels and features from the train set
    #print(data_subsets)
    #print(data_subsets[1])
    model = regression_model().fit(data_subsets[0], data_subsets[1])
    #predict the labels based on the training subset features
    y_train_pred = model.predict(data_subsets[0])
    #print(f'y train predict: {y_train_pred}')
    #predict the labels based on the validation subset featrues
    y_validation_pred = model.predict(data_subsets[2])
    #print(f'y_validation_pred = {y_validation_pred}')
    #predict the labels based on the test subset features
    y_test_pred = model.predict(data_subsets[4])
    # get the model parameters
    best_params = model.get_params()
    # calculate the performance metrics by comparing the prediction to actual scores
    metrics = return_model_performance_metrics(
        data_subsets[1], y_train_pred,
        data_subsets[3], y_validation_pred,
        data_subsets[5], y_test_pred
    )
    save_model(model, best_params, metrics, folder)
    return metrics

def tune_regression_model_hyperparameters(model, data_subsets, hyperparameters, folder):
    logging.info('Performing GridSearch with KFold...')
    model = model(random_state=13)
    kfold = KFold(n_splits=5, shuffle=True, random_state=13)
    clf = GridSearchCV(model, hyperparameters, cv=kfold)

    best_model = clf.fit(data_subsets[0], data_subsets[1])
    y_train_pred = best_model.predict(data_subsets[0])
    y_validation_pred = best_model.predict(data_subsets[2])
    y_test_pred = best_model.predict(data_subsets[4])

    best_params = best_model.best_params_
    metrics = return_model_performance_metrics(
        data_subsets[1], y_train_pred,
        data_subsets[3], y_validation_pred,
        data_subsets[5], y_test_pred
    )

    save_model(best_model, best_params, metrics, folder)
    return best_params, metrics
    
def evaluate_all_models(data_subsets):
    """Tunes the hyperparameters of SGDRegressor, DecisionTreeRegressor, RandomForestRegressor
        and XGBRegressor before saving the best model as a .joblib file, and
        best hyperparameters and performance metrics as .json files.
    """
    
    tune_regression_model_hyperparameters(
        DecisionTreeRegressor,
        data_subsets,
        dict(max_depth=list(range(1, 10))),
        '')

    tune_regression_model_hyperparameters(
        RandomForestRegressor,
        data_subsets,
        dict(
            n_estimators=list(range(80, 90)),
            max_depth=list(range(1, 10)),
            bootstrap=[True, False],
            max_samples = list(range(40, 50))),
        '')

    tune_regression_model_hyperparameters(
        xgb.XGBRegressor,
        data_subsets,
        dict(
            n_estimators=list(range(10, 30)),
            max_depth=list(range(1, 10)),
            min_child_weight=list(range(1, 5)),
            gamma=list(range(1, 3)),
            learning_rate=np.arange(0.1, 0.5, 0.1)),
        ''
    )




if __name__ == "__main__":
    df = pd.read_csv(Path('AirbnbDataSci/tabular_data/AirBnbData.csv'))
    df_2 = td.clean_tabular_data(df)
    features, labels = td.load_airbnb(df_2, "Price_Night")  
    split_data = split_the_data(features, labels, 0.7, 0.5)
    baseline_score = get_baseline_score(LinearRegression, split_data, 'models/regression')
    all_models = evaluate_all_models()







pd.options.mode.chained_assignment = None