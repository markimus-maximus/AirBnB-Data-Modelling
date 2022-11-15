from statistics import mode
from pathlib import Path
from scipy.sparse import data
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  GridSearchCV, KFold, train_test_split
from sklearn.metrics import  mean_absolute_error, mean_squared_error, r2_score

import glob
import joblib
import json
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import tabular_data as td
import time
import xgboost as xgb

pd.options.mode.chained_assignment = None

def return_model_performance_metrics(y_train, y_train_pred, y_validation, y_validation_pred, y_test, y_test_pred):
    """Calculates the RMSE and R2 score of a regression model.
    Args:
        y_train (array): Features for training.
        y_train_pred (array): Features predicted with training set.
        y_validation (array): Features for validation
        y_validation_pred (array): Features predicted with validation set.
        y_test (array): Features for testing.
        y_test_pred(array): Features predicted with testing set.
    Returns:
        metrics (dict): Training, validation and testing performance metrics.
    """
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

def return_multiple_score_metrics():
    json_list = []
    paths = glob.glob(r'models/regression/*/metrics.json')
    
    for path in paths:
        #print(path)
        model = path[18:-13]
        #print(f'model is {model}')
        with open(path, 'r') as file:
            text_data = file.read()
            text_data = '[' + re.sub(r'\}\s\{', '},{', text_data) + ']'
            json_data = json.loads(text_data)
            #json_data = [json.loads(line) for line in file]
            #for json_obj in file: 
                #dict = json.loads(json_obj)
                #json_list.append(json.loads(json_obj))
        print(json_data)

def save_model(model, hyperparameters, metrics, folder_for_files):
    """saves the information of a tuned regression model.
    Args:
        model (class): Saved as a .joblib file.
        hyperparameters (dict): Saved as a .json file.
        metrics (dict): Saved as a .json file.
        folder (str): The directory path of where to save the data.
    """
    joblib.dump(model, f'{folder_for_files}/model.joblib')
    with open(f'{folder_for_files}/hyperparameters.json', 'a') as outfile:
        json.dump(hyperparameters, outfile)
    with open(f'{folder_for_files}/metrics.json', 'a') as outfile:
        json.dump(metrics, outfile)

def split_the_data(features, labels, test_to_rest_ratio, validation_to_test_ratio, random_state):
    '''Splits data into train:validation:test subsets
    '''
    #split the data into test and train data; the 0.3 describes the data which is apportioned to the test set 
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_to_rest_ratio, random_state=random_state)
    #print(y_train)
    #resplit the test data again to get a final 15 % for both test and validation
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=validation_to_test_ratio, random_state=random_state)
    
    return [X_train, y_train, X_validation, y_validation, X_test, y_test]

def get_baseline_score(regression_model, data_subsets):
    """Tunes the hyperparameters of a regression model and saves the information.
    Args:
        model (class): The regression model to be as a baseline.
        sets (list): List in the form [X_train, y_train, X_validation,
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
    params = model.get_params()
    # calculate the performance metrics by comparing the prediction to actual scores
    metrics = return_model_performance_metrics(
        data_subsets[1], y_train_pred,
        data_subsets[3], y_validation_pred,
        data_subsets[5], y_test_pred
    )
    return model, metrics, params

def tune_regression_model_hyperparameters(model, data_subsets, hyperparameters):
    """Tunes the hyperparameters of a regression model and saves the information.
    Args:
        model (class): The regression model to be as a baseline.
        sets (list): List in the form [X_train, y_train, X_validation,
            y_validation, X_test, y_test].
        folder (str): The directory path of where to save the data.
    Returns:
        metrics (dict): Training, validation and testing performance metrics.
    """
    logging.info('Performing GridSearch with KFold...')
    model = model(random_state=13)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, hyperparameters, cv=kfold)
    # 
    model = grid_search.fit(data_subsets[0], data_subsets[1])
    y_train_pred = model.predict(data_subsets[0])
    y_validation_pred = model.predict(data_subsets[2])
    y_test_pred = model.predict(data_subsets[4])

    params = model.best_params_
    metrics = return_model_performance_metrics(
        data_subsets[1], y_train_pred,
        data_subsets[3], y_validation_pred,
        data_subsets[5], y_test_pred
    )
    return model, params, metrics
    
def evaluate_all_models(data_subsets):
    """Tunes the hyperparameters of DecisionTreeRegressor, RandomForestRegressor
        and XGBRegressor before saving the best model as a .joblib file, and
        best hyperparameters and performance metrics as .json files.
    """
    
    decision_tree = tune_regression_model_hyperparameters(
        DecisionTreeRegressor,
        data_subsets,
        dict(max_depth=list(range(1, 5))),
        )

    random_forest = tune_regression_model_hyperparameters(
        RandomForestRegressor,
        data_subsets,
        dict(
            #also decreased here fromm 100 to 90 to try to decrease overfitting
            n_estimators=list(range(75, 90)),
            #decreased here to remove overfitting
            max_depth=list(range(2, 4)),
            bootstrap=[True, False],
            #decrease from 55 to 50 to 40
            max_samples = list(range(30, 40))),
        )

    xgb_regressor =  tune_regression_model_hyperparameters(
        xgb.XGBRegressor,
        data_subsets,
        dict(
            n_estimators=list(range(20, 35)),
            #decreased max_depth to decrease overfitting
            max_depth=list(range(1, 3)),
            #changed to 2-6 to try to decrease overfitting
            min_child_weight=list(range(2, 6)),
            #pseudoregularisation strategy- high number = less complexity. Increased from 0 to 2 starting which inherently prunes the number of trees
            gamma=list(range(2, 3)),
            learning_rate=np.arange(0.05, 0.5, 0.05),
            )
    )
    combined_dictionaries_list = decision_tree + random_forest + xgb_regressor 
    return combined_dictionaries_list


def evaluate_models_multiple_times(num_iter, seed):
        BL_metrics_list_of_dicts = []
        DT_params_list_of_dicts = []
        DT_metrics_list_of_dicts = []
        RF_params_list_of_dicts = []
        RF_metrics_list_of_dicts = []
        XG_params_list_of_dicts = []
        XG_metrics_list_of_dicts = []

        np.random.seed(seed)
        random_states=  np.random.randint(99, size=num_iter)
        
        df = pd.read_csv(Path('AirbnbDataSci/tabular_data/AirBnbData.csv'))
        df_2 = td.clean_tabular_data(df)
        features, labels = td.load_airbnb(df_2, "Price_Night") 
        #increase training size due to overfitting of more complex models
        for iteration in random_states: 
            split_data = split_the_data(features, labels, 0.75, 0.5, random_state=iteration)
            
            BL_metrics, BL_params, BL_model = get_baseline_score(LinearRegression, split_data)
            BL_metrics_list_of_dicts.append(BL_metrics)
            models_list = evaluate_all_models(split_data)
            #print(models_list)
            DT_model = models_list[0]
            DT_params_list_of_dicts.append(models_list[1])
            DT_metrics_list_of_dicts.append(models_list[2])
            RF_model = models_list[3]
            RF_params_list_of_dicts.append(models_list[4])
            RF_metrics_list_of_dicts.append(models_list[5])
            XG_model = models_list[6]
            XG_params_list_of_dicts.append(models_list[7])
            XG_metrics_list_of_dicts.append(models_list[8])


        print(f'BL_metrics_list_of_dicts is {BL_metrics_list_of_dicts}')
        #print(DT_params_list_of_dicts)
        BL_metrics = get_aggregate_scores(BL_metrics_list_of_dicts)

        save_model(BL_model, BL_params, BL_metrics, Path(r"C:\Users\marko\DS Projects\AirBnB-Data-Modelling\main project file\models\regression\linear_regression"))
        
        DT_metrics = get_aggregate_scores(DT_metrics_list_of_dicts)
        
        #get the most common parameters of all of the iterations
        DT_common_params = mode([ sub['max_depth'] for sub in DT_params_list_of_dicts ])
        DT_common_params = {'max_depth' : DT_common_params}
        print(DT_common_params)
        save_model(DT_model, DT_common_params, DT_metrics, Path(r"C:\Users\marko\DS Projects\AirBnB-Data-Modelling\main project file\models\regression\decision_tree_regressor"))

        RF_metrics = get_aggregate_scores(RF_metrics_list_of_dicts)
        
        
        #get the most common parameters of all of the iterations
        RF_common_n_estimators = mode([ sub['n_estimators'] for sub in RF_params_list_of_dicts ])
        RF_common_max_depth = mode([ sub['max_depth'] for sub in RF_params_list_of_dicts ])
        RF_common_bootstrap =mode([ sub['bootstrap'] for sub in RF_params_list_of_dicts ])
        RF_common_max_samples =mode([ sub['max_samples'] for sub in RF_params_list_of_dicts ])
        RF_common_params = {'n_estimators': RF_common_n_estimators,
                            'max_depth' : RF_common_max_depth,
                            'bootstrap' : RF_common_bootstrap,
                            'max_samples': RF_common_max_samples}
        

        print(RF_common_params)
        save_model(RF_model, RF_common_params, RF_metrics, Path(r"C:\Users\marko\DS Projects\AirBnB-Data-Modelling\main project file\models\regression\random_forest_regressor"))

        XG_metrics = get_aggregate_scores(XG_metrics_list_of_dicts)
        
        #get the most common parameters of all of the iterations
        XG_common_n_estimators = mode([ sub['n_estimators'] for sub in XG_params_list_of_dicts ])
        XG_common_max_depth = mode([ sub['max_depth'] for sub in XG_params_list_of_dicts ])
        XG_common_min_child_weight = mode([ sub['min_child_weight'] for sub in XG_params_list_of_dicts ])
        XG_common_gamma =mode([ sub['gamma'] for sub in XG_params_list_of_dicts ])
        XG_common_learning_rate =mode([ sub['learning_rate'] for sub in XG_params_list_of_dicts ])
        XG_common_params = {'n_estimators' : XG_common_n_estimators,
                            'max_depth' : XG_common_max_depth,
                            'min_child_weight' : XG_common_min_child_weight,
                            'gamma': XG_common_gamma,
                            'learning_rate': XG_common_learning_rate}
        

        print(XG_common_params)
        
        save_model(XG_model, XG_common_params, XG_metrics, Path(r'C:\Users\marko\DS Projects\AirBnB-Data-Modelling\main project file\models\regression\xgboost_regressor'))
            
def get_aggregate_scores(list_of_dictionaries):
    list_of_train_rmse = [b['train_pred_rmse'] for b in list_of_dictionaries]

    train_mean_rmse = np.mean(list_of_train_rmse)
    train_std_rmse = np.std(list_of_train_rmse)
    train_var_rmse = (train_std_rmse / train_mean_rmse) * 100

    list_of_validation_rmse = [a['validation_rmse_pred'] for a in list_of_dictionaries]
    val_mean_rmse = np.mean(list_of_validation_rmse)
    val_std_rmse = np.std(list_of_validation_rmse)
    val_var_rmse = (val_std_rmse / val_mean_rmse) * 100

    list_of_test_rmse = [a['test_rmse_pred'] for a in list_of_dictionaries]
    test_mean_rmse = np.mean(list_of_test_rmse)
    test_std_rmse = np.std(list_of_test_rmse)
    test_var_rmse = (test_std_rmse / test_mean_rmse) * 100
    
    validation_rmse_accuracy = train_mean_rmse / val_mean_rmse 
    test_rmse_accuracy = train_mean_rmse / test_mean_rmse 
    # print(BL_mean_rmse)
    # print(BL_std_rmse)
    # print(BL_rmse_var)
    list_of_train_r2 = [d['train_r2_pred'] for d in list_of_dictionaries]
    list_of_validation_r2 = [d['validation_r2_pred'] for d in list_of_dictionaries]
    list_of_test_r2 = [d['test_r2_pred'] for d in list_of_dictionaries]
    train_mean_r2 = np.mean(list_of_train_r2)
    train_std_r2 = np.std(list_of_validation_r2)
    train_var_r2 = (train_std_r2 / train_mean_r2) * 100
    val_mean_r2 = np.mean(list_of_validation_r2)
    val_std_r2 = np.std(list_of_validation_r2)
    val_var_r2 = (val_std_r2 / val_mean_r2) * 100
    test_mean_r2 = np.mean(list_of_test_r2)
    test_std_r2 = np.std(list_of_validation_r2)
    test_var_r2 = (test_std_r2 / test_mean_r2) * 100
    metrics = { 'train_mean_rmse': train_mean_rmse,
                'train_std_rmse': train_std_rmse,
                'train_rmse_var': train_var_rmse,
                'train_mean_r2' : train_mean_r2,
                'train_std_r2' : train_std_r2,
                'train_var_r2' : train_var_r2,
                'val_mean_rmse': val_mean_rmse,
                'val_std_rmse': val_std_rmse,
                'val_rmse_var': val_var_rmse,
                'val_mean_r2': val_mean_r2,
                'val_std_r2': val_std_r2,
                'val_r2_var':val_var_r2,
                'test_mean_r2': test_mean_r2,
                'test_std_r2': test_std_r2,
                'test_var_r2': test_var_r2,
                'test_mean_rmse' : test_mean_rmse,
                'test_std_rmse' : test_std_rmse,
                'test_var_rmse' : test_var_rmse,
                'validation_rmse_accuracy': validation_rmse_accuracy,
                'test_rmse_accuracy' : test_rmse_accuracy
                }
    return metrics

def find_best_model():
    """Searches through the regression_models directory to find the model
        with the smallest RMSE value for the validation set (best model).
    Returns:
        best_model (class): Loads the model.joblib file.
        best_hyperparameters (dict): Loads the hyperparameters.json file.
        best_metrics (dict): Loads the metrics.json file.
    """
    logging.info('Finding best model...')

    paths = glob.glob(r'models/regression/*/metrics.json')
    rmse = {}
    for path in paths:
        print(f'path is {path}')
        model = path[18:-13]
        print(f'model is {model}')
        with open(path) as file:
            metrics = json.load(file)
        rmse[model] = metrics["val_mean_rmse"] 

    model_name = min(rmse, key=rmse.get)
    #print(f'model_name: {model_name}')
    model = joblib.load(f'models/regression/{model_name}/model.joblib')
    with open(f'models/regression/{model_name}/hyperparameters.json', 'rb') as file:
            hyperparameters = json.load(file)
    with open(f'models/regression/{model_name}/metrics.json', 'rb') as file:
            metrics = json.load(file)
    return model, hyperparameters, metrics


if __name__ == "__main__":
    evaluate_models_multiple_times(10, 2)
    #evaluate_models_multiple_times(5, 0)
    find_best_model()







pd.options.mode.chained_assignment = None