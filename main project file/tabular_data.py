from ast import literal_eval
from DataHandler import DataHandler
from IPython.display import display
from pathlib import Path
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 200



def remove_rows_with_missing_rating(dataframe_to_drop_rows, *column_names):
    """Removes all rows containing NaN values
    Args: 
        dataframe_to_drop_rows: the dataframe to drop rows containing NaN
        *column_names:  column names to exclude NaN
    Returns:
        Dataframe with rows dropped"""
    dropped_dataframe = dataframe_to_drop_rows.dropna(subset = column_names, axis=0)
    return(dropped_dataframe)


def combine_description_strings(dataframe_to_combine_strings, column_name):
    """Combines description data into continuous string of data, removing missing values and whitespace
    Args:
        dataframe_to_combine_strings: dataframe containing description data
        column_name: name of the column
    Returns:
        dataframe with complete descriptions"""
    #Convert the list into a list that pandas recognises
    dataframe_to_combine_strings[column_name] = dataframe_to_combine_strings[column_name].apply(lambda x: literal_eval(x))
    #Flatten the dataframe so that each list element becomes its own column
    dataframe_expanded = dataframe_to_combine_strings[column_name].apply(pd.Series)
    # Drop column containing repetitive 'About this space'
    dataframe_expanded = dataframe_expanded.drop(0, axis=1 )
    #Replace all NaN with ''
    dataframe_expanded = dataframe_expanded.fillna('')
    #Convert all data into str
    dataframe_expanded = dataframe_expanded.astype(str)
    #Combine the columns containing the different string list elements into one string
    dataframe_expanded = dataframe_expanded[[1 ,2 ,3, 4, 5, 6, 7, 8, 9]].apply(''.join, axis = 1)
    #Re-add the modified dataframe to the column
    dataframe_to_combine_strings[column_name] = dataframe_expanded
    return dataframe_to_combine_strings


def set_default_feature_values(dataframe_to_set_default_values, *columns):
    """Sets features with 1 when data is missing
    Args:
        dataframe_to_set_default_values: dataframe containing the data to be processed
        *columns: column names to apply the processing to"""
    columns = list(columns)
    dataframe_to_set_default_values[columns] = dataframe_to_set_default_values[columns].fillna(1)
    return(dataframe_to_set_default_values)
    #print(dataframe_to_set_default_values["guests", "beds", "bathrooms", "bedrooms"])


def clean_tabular_data(dataset):
    """ Prepares tabular data for inputing to model
    Args: 
    dataset: dataframe containing the data"""
    raw_dataframe = remove_rows_with_missing_rating(dataset, 'Description', 'Category', 'Cleanliness_rate','Accuracy_rate','Communication_rate','Location_rate','Check-in_rate','Value_rate')

    dataframe_combined_strings = combine_description_strings(raw_dataframe, 'Description')

    return set_default_feature_values(dataframe_combined_strings, "guests", "beds", "bathrooms", "bedrooms")


def load_airbnb(dataframe_input, label):
    """Generates AirBnB data for training classification model
    Args:
        dataframe_input: the dataframe containing the airbnb data
        label: the label correspodning to y dataset for model training
    Returns:
        dataframe excluding the label
        datafram column contining label data"""
    
    df = dataframe_input.drop(label, axis = 1)
    df = df.drop(['ID', 'Title','Description','Amenities','Location','url'], axis=1)
    return df, dataframe_input[label]
    
if __name__ == "__main__":
    dataset = DataHandler.csv_to_dataframe(Path('AirbnbDataSci/tabular_data/AirBnBData.csv'))
    print(dataset)
    cleaned_dataframe = clean_tabular_data(dataset)
    print(cleaned_dataframe)
    #cleaned_dataframe.to_csv('AirbnbDataSci/tabular_data/clean_csv')
   