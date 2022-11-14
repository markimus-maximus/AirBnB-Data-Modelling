from statistics import mode
from pathlib import Path
from scipy.sparse import data
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  GridSearchCV, KFold, train_test_split
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, r2_score
from tabulate import tabulate


import glob
import joblib
import json
import logging
import math
import matplotlib.pyplot as plt
import modelling as md
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import tabular_data as td
import time
import xgboost as xgb


def get_baseline_classification_score(model, data_subsets):
    """Tunes the hyperparameters of a regression model and saves the information.
    Args:
        model (class): The regression model to be as a baseline.
        sets (list): List in the form [X_train, y_train, X_val,
            y_val, X_test, y_test].
        folder (str): The directory path of where to save the data.
    Returns:
        metrics (dict): Training, validation and testing performance metrics.
    """
    logging.info('Calculating baseline score...')
    #fit the data baes on labels and features from the train set
    #print(data_subsets)
    #print(data_subsets[1])
    model = model(max_iter=1000).fit(data_subsets[0], data_subsets[1])
    #predict the labels based on the training subset features
    y_train_pred = model.predict(data_subsets[0])
    cfm_train = create_confusion_matrix(data_subsets[1], y_train_pred)
    print(cfm_train)
    #predict the labels based on the val subset featrues
    y_val_pred = model.predict(data_subsets[2])
    cfm_val = create_confusion_matrix(data_subsets[3], y_val_pred)
    print(cfm_val)
    #predict the labels based on the test subset features
    y_test_pred = model.predict(data_subsets[4])
    cfm_test = create_confusion_matrix(data_subsets[5], y_test_pred)
    print(cfm_test)
    #get the model parameters
    params = model.get_params()
    # calculate the performance metrics by comparing the prediction to actual scores
    
    metrics = get_all_data_metrics(data_subsets, y_train_pred, y_val_pred, y_test_pred)

    return model, params, metrics 

def get_all_data_metrics(data_subsets, train_pred, val_pred, test_pred):
    train_prec_score, train_rec_score, train_f1_score = get_performance_metrics(data_subsets[1], train_pred)
    val_prec_score, val_rec_score, val_f1_score =   get_performance_metrics(data_subsets[3], val_pred)
    test_prec_score, test_rec_score, test_f1_score = get_performance_metrics(data_subsets[5], test_pred)
                                                            
   
    metrics = { "train_prec_score": train_prec_score, 
                "train_rec_score": train_rec_score,
                "train_f1_score": train_f1_score,
                "val_prec_score": val_prec_score, 
                "val_rec_score": val_rec_score,
                "val_f1_score": val_f1_score,
                "test_prec_score": test_prec_score, 
                "test_rec_score": test_rec_score,
                "test_f1_score": test_f1_score}   
    
    return metrics

def create_confusion_matrix(true_labels, pred_labels):
    
    cf_matrix = confusion_matrix(true_labels, pred_labels)
    return cf_matrix / cf_matrix.sum()

def visualise_confusion_matrix(confusion_matrix): 
    display = ConfusionMatrixDisplay(confusion_matrix)
    display.plot()
    plt.show()

def get_confusion_matrix(true_labels, pred_labels):
    cf = create_confusion_matrix(true_labels, pred_labels)
    return visualise_confusion_matrix(cf)

def get_performance_metrics(true_labels, pred_labels):
    prec_score = precision_score(true_labels, pred_labels, average='macro')
    rec_score = recall_score(true_labels, pred_labels, average='macro')
    f1_scr = f1_score(true_labels, pred_labels, average='macro')
    return prec_score, rec_score, f1_scr

if __name__ == "__main__":
    df = pd.read_csv(Path('AirbnbDataSci/tabular_data/AirBnbData.csv'))
    print(df)
    df_2 = td.clean_tabular_data(df)
    print(df_2)


    features, labels = td.load_airbnb(df_2, "Category")
    print(features)

    split_data = md.split_the_data(features, labels, 0.75, 0.5, 42)
    print(split_data)
    BL = get_baseline_classification_score(LogisticRegression, split_data)
    print(BL)


#get_performance_metrics()

