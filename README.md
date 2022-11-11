# AirBnB-Data-Modelling

# Data Science Project- Modelling AirBnB data.

My fourth project with the aiCore workflow was to design and implement range of models to characterise both tabular and image data obtained from AirBnB.

The milestones in this project:  

- Before starting a virtual environment was set up to portably deploy required dependencies

- Data preparation

- Create a regression model

- Create a classification model

- Create a configurable neural network

- Reuse the frame work for another use-case with the AirBnB data

## Data preparation

In order to feed the AirBnB data into the model, a number of functions were written to process both tabular and image data.

### Tabular data preparation

The tabular data contained the following data categories:

The tabular dataset has the following column:

- ID: Unique identifier for the listing
- Category: The category of the listing
- Title: The title of the listing
- Description: The description of the listing
- Amenities: The available amenities of the listing
- Location: The location of the listing
- guests: The number of guests that can be accommodated in the listing
- beds: The number of available beds in the listing
- bathrooms: The number of bathrooms in the listing
- Price_Night: The price per night of the listing
- Cleanliness_rate: The cleanliness rating of the listing
- Accuracy_rate: How accurate the description of the listing is, as reported by previous guests
- Location_rate: The rating of the location of the listing
- Check-in_rate: The rating of check-in process given by the host
- Value_rate: The rating of value given by the host
- amenities_count: The number of amenities in the listing
- url: The URL of the listing
- bedrooms: The number of bedrooms in the listing

tabular_data.py contains the code written to process the tabular data.  

To begin, the data contained many rows with missing ratings data. Since these ratings will be used to train the model, any missing data can be excluded with `remove_rows_with_missing_rating(dataframe_to_drop_rows, *column_names)`, which takes the dataframe and variable numbers of column names to delete rows containing `NaN` using `pd.dropna()`.

The 'Description' column contains a list of strings together make up the description, and accordingly the `combine_description_strings(dataframe_to_combine_strings, column_name)` function was written. Unfortunately, pandas did not recognise the descriptions as lists of strings, and as such a means of separating out the list elements was required. To achieve this, firstly the Description column was converted to a recognisable list with the following code:
`dataframe_to_combine_strings['Description'] = dataframe_to_combine_strings['Description'].apply(lambda x: literal_eval(x))`
After conversion to a recognisable list, the data were flattened such that each component of the list was contained by a distinct row with `apply(pd.Series)`. This flattening of the data allows for row-specific functions to be applied each list row. Since there were lots of NaN values in the list, these were replaced to contain no data at all, this removing inappropriate blank spaces from the data. A repetitive and unnecessary 'About this space' string preceded the description as a distinct list element and accordingly this element was sliced out from every row. After the list-based functions had been carried out, the flattened list was recombined to a solitary column which was a single string with the following code:
`dataframe_expanded = dataframe_expanded[[1 ,2 ,3, 4, 5, 6, 7, 8, 9]].apply(''.join, axis = 1)`
The number of bedrooms and bathrooms in the data sometimes did not contain data, and as such these missing values needed to be addressed. It was assumed that if there were no data in these columns then the actual number was 1. This was coded for with the `set_default_feature_values(dataframe_to_set_default_values, *columns)` function and using `pd.fillna(1)`.
The above 3 functions were encompassed into the `clean_tabular_data(dataset)` function to be called when the data is to be cleaned.
To prepare the data for addition into the model, a series of tuples containing the feature (the column headings) and the label (row values in the column) were generated with the `load_airbnb(dataframe_input, label, *columns_to_exclude)` function. Within this function, any unwanted columns (non-numerical data) can be excluded before the dataset is prepared.

### Image data preparation
To ensure each image was the same size, `resize_images(normalised_height:int, original_image_folder_directory, directory_for_resized_images:str)` was written using the `Pillow` library. This function  loads each image from different subfolders and resizes it to the same height and width, before saving the new version in a new `processed_images` folder. The aspect ratio of the image was calculated and the width was adjusted proportionally to the change in height. In order to characterise the data to identify the smallest height (with which to normalise the other images to) the
`get_lowest_image_dimensions_from_folder(original_image_folder_directory:str)` function was written. This function returns the lowest width and height of a collection of images in separate subfolders.

## Creating a regression model

Data prepeared previously was used to train the models below. The file `modelling.py` was created to contain the code related to linear regression. 

### Creating a function for splitting data

In order to ensure that model performance is assessed on "unseen" data, the data were first split into 3 data sets; train, validation, split, in proportions of 70:15:15, respectively. The function `split_the_data(features, labels, test_to_rest_ratio, validation_to_test_ratio, random_state)` was written to take the desired ratios of train:test and validation:test ratios (in this example 0.7 and 0.5, respectively), as well as the pseudo-random state of the data shuffling, a feature which is important to remain consistent between models to ensure fair comparisons.  The splitting of data was carried out with the `train_test_split` method from `sklearn`. The function returned split data in 6 lists: 
`[X_train, y_train, X_validation, y_validation, X_test, y_test]`

### Creating a simple regression model as baseline

As a baseline for comparison, a linear regression model was created. The function `get_baseline_score(regression_model, data_subsets, folder)` was created to take the split data (above) to train the model, and a folder to later output metrics.

### Create a function for multiple iterations of the modelling process and calculate average metrics and most-common parameters

An unrepresentative model may be generated when fitting a model to one pseudo-random subset of data. A common approach to gain a better-rounded model is to  is to create multiple subsets of pseudo-random data with which to train the model. In order to carry out this need, the function 

### Evaluate the regression model performance

### Implement a custom function to tune the hyperparameters of the model

### Tune the hpperparameters of the modell using methods from SKLearn

### Saving the model

### Beating the baseline regression model

### Comparing the models and finding the best


