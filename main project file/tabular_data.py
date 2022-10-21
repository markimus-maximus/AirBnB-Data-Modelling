from ast import literal_eval
from DataHandler import DataHandler
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 200
from pathlib import Path



def remove_rows_with_missing_rating(dataframe_to_drop_rows, subset = None):
    dropped_dataframe = dataframe_to_drop_rows.dropna(subset = subset, axis=0)
    return(dropped_dataframe)

def combine_description_strings(dataframe_to_combine_strings, subset = None):
    #First remove row with no description at all
    dropped_dataframe = dataframe_to_combine_strings.dropna(subset = subset, axis=0)
    #Convert the list into a list that pandas recognises
    dataframe_to_combine_strings['Description'] = dropped_dataframe['Description'].apply(lambda x: literal_eval(x))
    #Flatten the dataframe so that each list element becomes its own column
    dataframe_expanded = dataframe_to_combine_strings["Description"].apply(pd.Series)
    # Drop column containing repetitive 'About this space'
    dataframe_expanded = dataframe_expanded.drop(0, axis=1 )
    #Replace all NaN with ''
    dataframe_expanded = dataframe_expanded.fillna('')
    #Convert all data into str
    dataframe_expanded = dataframe_expanded.astype(str)
    #Combine the columns containing the different string list elements into one string
    dataframe_expanded = dataframe_expanded[[1 ,2 ,3, 4, 5, 6, 7, 8, 9]].apply(''.join, axis = 1)
    #Re-add the modified dataframe to the column
    dataframe_to_combine_strings['Description'] = dataframe_expanded
    return dataframe_to_combine_strings

def set_default_feature_values(dataframe_to_set_default_values):
    dataframe_to_set_default_values[["guests", "beds", "bathrooms", "bedrooms"]] = dataframe_to_set_default_values[["guests", "beds", "bathrooms", "bedrooms"]].fillna(1)
    return(dataframe_to_set_default_values)
    #print(dataframe_to_set_default_values["guests", "beds", "bathrooms", "bedrooms"])

def clean_tabular_data():

    raw_dataframe = remove_rows_with_missing_rating(dataset, subset='Value_rate')

    dataframe_combined_strings = combine_description_strings(raw_dataframe, subset = 'Description')

    return set_default_feature_values(dataframe_combined_strings)
    
if __name__ == "__main__":
    dataset = DataHandler.csv_to_dataframe(Path('AirbnbDataSci/tabular_data/AirBnBData.csv'))
    cleaned_dataframe = clean_tabular_data()
    cleaned_dataframe.to_csv('AirbnbDataSci/tabular_data/clean_tabular_data.csv')