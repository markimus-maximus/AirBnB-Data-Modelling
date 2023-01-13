import joblib
import json
import numpy as np
import os
import pandas as pd
import plotly.express as px
import tabular_data as td
from time import time, time_ns
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision.datasets as datasets
import yaml

from datetime import datetime
from IPython.display import display
from pathlib import Path
from sklearn.model_selection import train_test_split
from time import time
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch import tensor
  



# def data_prep(path_to_csv):
#     data = pd.read_csv(path_to_csv)
#     #git commit 'maindrop non-numerical
#     features_dropped = data.drop(['Category', 'ID', 'Title','Description','Amenities','Location','url'], axis=1)
#     #index only the label and covert to tensor of floats
#     label_all = tensor(features_dropped['Price_Night']).float()
#     #index features by dropping label and covert to tensor of floats
#     features_all = tensor(features_dropped.drop('Price_Night', axis=1).values).float()  
#     number_of_features = features_all.shape[1]
#     print(f'number of features: {number_of_features}')
#     return features_all, label_all

#class to make the data iterable by index, reurning tuple of tensor features, tensor label
class AirbnbNightlyPriceDataset(Dataset):
    """Creates a data class to allow a given dataset to be iterable for batch feeding the model.
    Returns:
        Indexable dataset with shape
        """
    def __init__(self, features_all, label_all):
        assert len(features_all) == len(label_all), "Features and labels must be of equal length."
        #initialise parent class
        super().__init__()  
        self.features_all = features_all 
        self.label_all  = label_all 
    # describes behaviour when the data in the object are indexed
    def __getitem__(self, index):
        
        #index the features and labels 
        return self.features_all[index], self.label_all[index]        

    # describes behaviour when len is called on the object
    def __len__(self):
        return self.features_all.shape[0]

def split_dataset(dataset, labels:str, random_state:int):
    """Splits up the dataset into training, validation and testing datasets
    in the repective ratio of 60:20:20.
    Args:
        dataset (DataFrame): The dataset to be split.
        targets (list): A list of columns to be used as the targets.
        random_state (int): The random state used in the split.
    Returns:
        (tuple): (X_train, y_train, X_validation, y_validation, X_test, y_test)
            in tensor form.
    """
    dataset_numerical_only = dataset.drop(['Category', 'ID', 'Title','Description','Amenities','Location','url'], axis=1).reset_index(drop=True)
    # dataset_numerical_only = dataset_numerical_only.
    X = torch.tensor(dataset_numerical_only.drop(['Price_Night'], axis=1).values).float()
    print(X.shape)
    y = torch.tensor(dataset_numerical_only[labels].values).float()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=random_state)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.5, random_state=random_state)
    return (X_train, y_train, X_validation, y_validation, X_test, y_test)

def create_dataloader(X_train:tensor, y_train:tensor, batch_size:int):
    """Creates a DataLoader from the training dataset.
    Args:
        X_train (tensor): The tensor containing the training features.
        y_train (tensor): The tensor containing the training targets.
        batch_size (int: The size of one batch.
    Returns:
        dataloader (class): DataLoader created from the training set.
    """
    train_dataset = AirbnbNightlyPriceDataset(X_train, y_train)
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

class LinearRegression(torch.nn.Module):
    """A linear regression model using neural network architecture.
    Args:
        num_features: Number of features for training the model
        batch_size: The size of the batches used to train the model
    Returns:
        Neural network architecture to be trained"""
    #initialise the parameters
    def __init__(self, num_features, batch_size):
        super().__init__()
        
        self.linear_layer = torch.nn.Linear(num_features, batch_size) # passing the batch size and the number of features

    #describes behaviour when the class is called, similar to __call__
    def forward(self, features):
        #use the layers to process the features, returning a prediction
        return self.linear_layer(features)

# model = LinearRegression()

def train_linear_regression(model, num_epochs:int, dataloader):
    """Trains linear model with neural network architecture.
    Args:
        model: a neural nework model instance
        num_epochs: The number of epochs to train the model on
        dataloader: A dataloader instance for feeding the model
        """
    #The algorithm for this optimisation step is stochastic gradient descent (since linear regression is used to derive the model). Passing model.parameters and the learning rate lr
    optimiser = torch.optim.SGD(model.parameters(), lr = 0.001)

    for epoch in range(num_epochs):
        #iterates in batches for the number of epochs defined
        for batch in dataloader:
            #print(batch)
            #unpack the batch into features and labels
            features, label = batch
            #make a prediction from the model based on a batch of features
            prediction = model(features)
            print(f'prediciton shape = {prediction.shape}')
            #calculate the loss- used to apply to gradient descent algorithm. Using mse as this is a linear regression problem
            loss = F.mse_loss(prediction, label)
            print(f'loss: {loss}')
            #populates gradients of model parameters with respect to loss
            diff = loss.backward()
            #print(f'diff: {diff}')
            #opimisation step 
            optimiser.step()
            #the .grad associated with tensor .backward does not go back to 0 with every iteration (clearly this would cause issues with SGD), accordingly must re-zero the grad of the optimiser after each iteration. But don't do this before .step!
            optimiser.zero_grad()

class NN(torch.nn.Module):
    """A neural network building class.
    Args: 
        input_dim: Input dimensions informing the number of nodes 
        hidden_dim_array: Dimensions of hidden array in a list of 2 integers
        output_dim: Output dimesnions
    Returns:
        Neural network architecture to be trained
    """
    def __init__(self, input_dim, hidden_dim_array, output_dim):
        super().__init__()
        self.hidden_dim_array = hidden_dim_array
        #define layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim_array[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_array[0], hidden_dim_array[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_array[1], output_dim)
        )
    #define how to stack the different elements of the network together
    def forward(self, X):
        return self.layers(X)

def get_nn_config(path_to_yaml:str):
    """Retrieves a neural network yaml file
    Args:
        Path to yaml file
    Returns:
        yaml data"""
    with open(path_to_yaml) as file:
        yaml_data= yaml.safe_load(file)
        print(yaml_data)
        return yaml_data

def train_nn(model, num_epochs:int, name_writer:str, dataloader,  hyperparameter_dict:dict, optimiser):
    """Trains a feed-forward neural network model
    Args:  
        model: A neural nework model instance
        num_epochs: The number of epochs to train the model on (int)
        name_writer: A name for Tensorboard graph (string)
        dataloader: A dataloader instance for feeding the model
        hyperparameter_dict: Dictionary containing hyperparameters (dict)
        optimiser: Type of optimiser to use
    Returns:
        training_metrics: A dictionary containing training_duration and training_loss}
        model_parameters: A dictionary containing optimiser_parameters, model state dictionary and batch size
    """
    #The algorithm for this optimisation step is stochastic gradient descent (since linear regression is used to derive the model). Passing model.parameters and the learning rate lr
    optimiser = optimiser(model.parameters(), lr = hyperparameter_dict['learning_rate'])

    writer = SummaryWriter()

    batch_idx = 0 # created this as a counter for writer since the batch resets within the loop

    #getting start time for training duration

    start_time=time()

    for epoch in range(num_epochs):
        #iterates in batches for the number of epochs defined
        for batch in dataloader:
            #print(batch)
            #unpack the batch into features and labels
            features, label = batch
            #make a prediction from the model based on a batch of features
            prediction = model(features)
            print(f'prediciton shape = {prediction.shape}')
            #calculate the loss- used to apply to gradient descent algorithm. Using mse as this is a linear regression problem
            training_loss = F.mse_loss(prediction, label)
            print(f'training_loss: {training_loss}')
            #populates gradients of model parameters with respect to loss
            diff = training_loss.backward()
            #print(f'diff: {diff}')
            #opimisation step 
            optimiser.step()
            #the .grad associated with tensor .backward does not go back to 0 with every iteration (clearly this would cause issues with SGD), accordingly must re-zero the grad of the optimiser after each iteration. But don't do this before .step!
            optimiser.zero_grad()
            writer.add_scalar(name_writer, training_loss.item(), batch_idx)
            batch_idx += 1

    end_time = time()
    training_duration = end_time - start_time
    #print(f'time for model training: {training_duration}')

    optimiser_parameters = optimiser.state_dict()
    #print(f'optimiser parameters: {optimiser_parameters}')

    model_state_dict = model.state_dict()
    #print(f'model state dict: {model_state_dict}')

    batch_size = {'batch_size': len(batch)}

    model_parameters = optimiser_parameters | model_state_dict | batch_size

    model_parameters = model_parameters.items()
    
    training_metrics = {'training_duration': training_duration, 'loss':training_loss}
    return training_metrics, model_parameters

def evaluate_model(model, dataloader_val, dataloader_test=None):
    """Evaluates neural network performance
    Args:
        model: neural network model instance
        dataloader_val: Dataloder instance for the validation dataset
        dataloader_test(optional): Dataloder instance for the test dataset
    Returns: Dictionary of performance metrics including: validation loss (mse), validation r2, test loss (mse), test r2, mean inference latency(ms)
        """
    model.eval()
    #we don't need the grad function here so switching it off with no_grad

    inference_latencies_list = []

    with torch.no_grad():
        for X, y in dataloader_val:
            #starting the timer for inference latency
            inference_latency_start = time_ns()
            #making the prediction
            prediction = model(X)
            #appending the latency into a list
            inference_latencies_list.append(time_ns() - inference_latency_start)
            val_loss = F.mse_loss(prediction, y)
            print(f'val_loss: {val_loss}')
            #need to instantiate r2 for this to work
            r2_inst = torchmetrics.R2Score()
            #need same dimensions, so flattenned prediction which was (x, 1) dimension
            #prediction_flattened = torch.flatten(prediction)
            val_r2 = r2_inst(prediction, y)
            
            print(f'val_r2: {val_r2}')

    if dataloader_test == None:
        test_loss, test_r2 = 'n/a', 'n/a'
           
    #we don't need the grad function here so switching it off with no_grad
    else: 
        with torch.no_grad():
            for X, y in dataloader_test:
                inference_latency_start = time()
                prediction = model(X)
                inference_latencies_list.append(time() - inference_latency_start)
                test_loss = F.mse_loss(prediction, y)
                print(f'test_loss: {test_loss}')
                #need to instantiate r2 for this to work
                r2_inst = torchmetrics.R2Score()
                #need same dimensions, so flattenned prediction which was (x, 1) dimension
                prediction_flattened = torch.flatten(prediction)
                test_r2 = r2_inst(prediction_flattened, y)
                print(f'test_r2: {test_r2}')

    mean_inference_latency = np.mean(inference_latencies_list)
    
    performance_metrics = {'val_loss':val_loss, 'val_r2':val_r2, 'test_loss':test_loss, 'test_r2':test_r2, 'mean_inference_latency':mean_inference_latency}

    return performance_metrics


class CNN(torch.nn.Module):
    """Generates architecture for convoluted neural network
    Returns:
        """
    def __init__(self):
        super().__init__()
        #define layers
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 7),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 7),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            #guess the value for th inputs in the first instance, when it's wrong pytorch will tell you the correct number
            torch.nn.Linear(4096, 10),
            torch.nn.Softmax()
        )
    #define how to stack the different elements of the network together
    def forward(self, X):
        return self.layers(X)

def train_cnn(model, num_epochs:int, name_writer:str, dataloader):
    """Trains a neural network model
    Args:  
        model: A neural nework model instance
        num_epochs: The number of epochs to train the model on (int)
        name_writer: A name for Tensorboard graph (string)
        dataloader: A dataloader instance for feeding the model
    Returns:
        training_metrics: A dictionary containing training_duration and training_loss}
        model_parameters: A dictionary containing optimiser_parameters, model state dictionary and batch size"""

    #The algorithm for this optimisation step is stochastic gradient descent (since linear regression is used to derive the model). Passing model.parameters and the learning rate lr
    optimiser = torch.optim.SGD(model.parameters(), lr = 0.001)

    writer = SummaryWriter()

    batch_idx = 0 # created this as a counter for writer since the batch resets within the loop

    start_time = time()
    

    for epoch in range(num_epochs):
        #iterates in batches for the number of epochs defined
        for batch in dataloader:
            #print(batch)
            #unpack the batch into features and labels
            features, label = batch
            #make a prediction from the model based on a batch of features
            prediction = model(features)
            # print(f'prediciton shape = {prediction.shape}')
            # print(f'label_shape = {label.shape}')
            #calculate the loss- used to apply to gradient descent algorithm. Using mse as this is a linear regression problem
            training_loss = F.cross_entropy(prediction, label)
            print(f'training_loss: {training_loss}')
            #populates gradients of model parameters with respect to loss
            diff = training_loss.backward()
            #print(f'diff: {diff}')
            #opimisation step 
            optimiser.step()
            #the .grad associated with tensor .backward does not go back to 0 with every iteration (clearly this would cause issues with SGD), accordingly must re-zero the grad of the optimiser after each iteration. But don't do this before .step!
            optimiser.zero_grad()
            writer.add_scalar(name_writer, training_loss.item(), batch_idx)
            batch_idx += 1

    end_time = time()
    training_duration = end_time - start_time
    #print(f'time for model training: {training_duration}')

    optimiser_parameters = optimiser.state_dict()
    #print(f'optimiser parameters: {optimiser_parameters}')

    model_state_dict = model.state_dict()
    #print(f'model state dict: {model_state_dict}')

    batch_size = {'batch_size': len(batch)}

    model_parameters = optimiser_parameters | model_state_dict | batch_size

    model_parameters = model_parameters.items()
    
    training_metrics = {'training_duration': training_duration, 'loss':training_loss}
    return training_metrics, model_parameters

def save_nn_model(folder_path:str, model, training_metrics:dict, performance_metrics:dict, all_parameters:dict, hyperparameters:dict):
    """Saves neural network model
    Args:
        folder_path: Path to the folder to contain the model
        model: Instance of the neural network to be saved
        performance_metrics: Dictionary of performance metrics
        all_parameters: Dictionary of model parameters
        hyperparametrs: Dictionary of all hyperparameters used to train the model
    Returns: 
        path_date: The time stamp that the model was saved with
        """
    today = datetime.now()
    today = today.strftime('\%Y-%m-%d_%H%M%S')
    print(today)
    os_dir = str(Path(folder_path))
    print(os_dir)
    path_date = str(os_dir + today)
    print(path_date)
    os.mkdir(path_date)
    print(os_dir)
    file_name = str(path_date) + '.pt'
    print(file_name)
    #saving the model
    torch.save(model.state_dict(), file_name)
    #need to convert from tensor to string of json won't save Tensor dtype
    with open(f'{path_date}/parameters.json', 'a') as outfile:
        all_parameters = str(all_parameters)
        json.dump(all_parameters, outfile)
    with open(f'{path_date}/training_metrics.json', 'a') as outfile:
        training_metrics = str(training_metrics)
        json.dump(training_metrics, outfile)
    with open(f'{path_date}/performance_metrics.json', 'a') as outfile:
        performance_metrics =  str(performance_metrics)
        json.dump(performance_metrics, outfile)
    with open(f'{path_date}/hyperparameters.json', 'a') as outfile:
        hyperparameters = str(hyperparameters)
        json.dump(hyperparameters, outfile)

    return path_date

def generate_nn_config():
    """Creates multiple configuration dictionaries containing hyperparameters for 
    the optimiser, learning rate and depth and width of the hidden
    layers.
    Returns:
        config_dict (dict): Dictionary containing multiple config dictionaries.
    """
    #learning rate values
    learning_rate_tests = [1e-4, 1e-5, 1e-6]
    #hidden dimension arrays
    hidden_dim_array_tests = [
    [2, 1], [4, 1], [6, 1], [8, 1], [16, 1],
     [2, 2], [2, 4], [2, 8], [2, 16], [4, 4], [4, 8]
    ]
    #declaring the choice of optimisers
    optimiser_options= [torch.optim.SGD, torch.optim.Adam]
    config_dict = {}
    counter = 0
    #generating the dictionary
    for learning_rate in learning_rate_tests:
        for hidden_dim_array in hidden_dim_array_tests:
            for optimiser in optimiser_options:
                config_dict[counter] = {}
                config_dict[counter]['optimiser'] = optimiser
                config_dict[counter]['learning_rate'] = learning_rate
                config_dict[counter]['hidden_dim_array'] = hidden_dim_array

                counter += 1
    #print(f'config_dict: {config_dict}')
    return config_dict

def find_best_nn(data_directory, input_dim:int, output_dim:int, name_writer:str, hyperparameter_dictionary:dict, folder_for_files:str):
    """Generates several instances of a model and automatically identifies the best neural network based on the model loss
    Args:
        data_directory: Directory for the data to feed into the model 
        input_dim: Dimensions informing the number of nodes for the input layer
        output_dim: Dimensions informing the number of nodes for the output layer
        name_writer: A name for Tensorboard graph (str)
        hyperparameter_dictionary: Dictionary containing hyperparameters (dict)
        folder_for_files: Directory for storing all files
    Returns: 
        The identity of the best model

        """
    dataset = pd.read_csv(data_directory)
    X_train, y_train, X_val, y_val, X_test, y_test =  split_dataset(dataset, ['Price_Night'], 42)
    
    num_epochs = get_num_epochs(1000, 100, dataset)

    dataloader_train = create_dataloader(X_train, y_train, batch_size = 100)

    loss_list = []

    best_model = str
     #iterating though the hyperparameter dictionary by each dictionary of hyperparameters  
    for dictionary in hyperparameter_dictionary.values():
        #print(f'dictionary_looks_like_this: {dictionary}')
        # print(dictionary['hidden_dim_array'])
        #creating model instance
        model = NN(dictionary['hidden_dim_array'], input_dim, output_dim)
        #training the model and returning the training metrics and model parameters
        training_metrics, all_parameters = train_nn(model, num_epochs, name_writer, dataloader_train,  dictionary, dictionary['optimiser'])
        #creating a dataloader for the validation set
        dataloader_val = create_dataloader(X_val, y_val, len(X_val))
        #evaluating the model's performance with the validation dataset
        evaluation_metrics = evaluate_model(model, dataloader_val, dataloader_test=None)
        print(f'eval_looks_like_this: {evaluation_metrics}')
        #getting the validation loss for this iteration of the model
        loss_this_iteration = np.array(evaluation_metrics['val_loss'])
        #appending loss from this iteration to the whole list of losses
        loss_list.append(loss_this_iteration)
        #saves the current model and returns a path which can be saved
        path = save_nn_model(folder_for_files, model, training_metrics, evaluation_metrics, all_parameters, dictionary)
        print(f'los_list: {loss_list}')
        # checks if the loss of this iteration beats the other iterations
        if loss_this_iteration < all(loss_list):
            print(f'new_best_model = {path}')
            best_model = path

    return best_model


def get_num_epochs(n_iters:int, batch_size:int, X_train):
    """Calculate number of passes through the entire training set.
    Args:
        n_iters (int): The number of batches iterated over.
        batch_size (int): The size of one batch.
        X_train (tensor): The tensor containing the training features.
    Returns:
        (int): The number of passes through the entire dataset.
    """
    return int(n_iters / (len(X_train) / batch_size))




if __name__ == '__main__':
    # dataset = pd.read_csv(r'AirbnbDataSci\tabular_data\clean_tabular_data.csv')
    # X_train, y_train, X_val, y_val, X_test, y_test =  split_dataset(dataset, ['Price_Night'], 42)
    
    


    # dataloader_train = create_dataloader(X_train, y_train, batch_size)
   
    hyperparameter_dictionary = generate_nn_config()
    # print(hyperparameter_dictionary)
    print(find_best_nn(r'AirbnbDataSci\tabular_data\clean_tabular_data.csv', 11, 1, 'model_optimisation', hyperparameter_dictionary, r'neural_networks\regression\test'))
    # training_metrics, model_parameters = train_nn(model_bl, num_epochs, 'testing', dataloader_train)
    # dataloader_val = create_dataloader(X_val, y_val, len(X_val))
    # evaluation_metrics = evaluate_model(model_bl, dataloader_val, test_dataloader=None)
    # #need to convert to string here or json dump incompatible with tensors
    # metrics = str(training_metrics | evaluation_metrics)
    # print(metrics)
    # print(model_parameters)
    # save_nn_model(r'neural_networks\regression', model_bl, metrics, model_parameters)
    # #print(model_parameters)

    #print(generate_nn_config())