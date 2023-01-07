import joblib
import json
import numpy as np
import os
import pandas as pd
import plotly.express as px
import tabular_data as td
from time import time
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
#     #drop non-numerical
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

def split_dataset(dataset, labels, random_state):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)
    return (X_train, y_train, X_validation, y_validation, X_test, y_test)

def create_dataloader(X_train, y_train, batch_size):
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
    
    #initialise the parameters
    def __init__(self, num_features, batch_size):
        super().__init__()
        
        self.linear_layer = torch.nn.Linear(num_features, batch_size) # passing the batch size and the number of features

    #describes behaviour when the class is called, similar to __call__
    def forward(self, features):
        #use the layers to process the features, returning a prediction
        return self.linear_layer(features)

# model = LinearRegression()

def train_linear_regression(model, num_epochs):
    
    #The algorithm for this optimisation step is stochastic gradient descent (since linear regression is used to derive the model). Passing model.parameters and the learning rate lr
    optimiser = torch.optim.SGD(model.parameters(), lr = 0.001)

    for epoch in range(num_epochs):
        #iterates in batches for the number of epochs defined
        for batch in dataloader_train:
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
    def __init__(self):
        super().__init__()
        #define layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(11, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )
    #define how to stack the different elements of the network together
    def forward(self, X):
        return self.layers(X)

def get_nn_config(path_to_yaml):
    with open(path_to_yaml) as file:
        yaml_data= yaml.safe_load(file)
        print(yaml_data)
        return yaml_data

def generate_nn_config():
    """Creates multiple configuration dictionaries containing info on 
    the optimiser, learning rate and depth and width of the hidden
    layers.
    Returns:
        config_dict (dict): Containing multiple config dictionaries.
    """
    learning_rate_tests = [1e-4, 1e-5, 1e-6]
    hidden_dim_array_tests = [
        [2], [4], [6], [8], [10],
        [2, 2], [4, 4], [6, 6], [8, 8], [10, 10],
        [2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8], [10, 10, 10],
        [8, 4], [8, 6], [8, 6], [6, 4], [6, 2], [4, 2],
        [8, 6, 4], [10, 6, 4], [10, 6, 2], [8, 4, 2], [6, 4, 2],
        [10, 8, 6, 4],
        [10, 8, 6, 4, 2],
    ]
    config_dict = {}
    counter = 0
    for learning_rate in learning_rate_tests:
        for hidden_dim_array in hidden_dim_array_tests:
            config_dict[counter] = {}
            config_dict[counter]['optimiser'] = torch.optim.SGD
            config_dict[counter]['learning_rate'] = learning_rate
            config_dict[counter]['hidden_dim_array'] = hidden_dim_array
            counter += 1
    print(f'config_dict: {config_dict}')
    return config_dict




def train_simple(model, num_epochs, name_writer, dataloader):

    #The algorithm for this optimisation step is stochastic gradient descent (since linear regression is used to derive the model). Passing model.parameters and the learning rate lr
    optimiser = torch.optim.SGD(model.parameters(), lr = 0.001)

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
            loss = F.mse_loss(prediction, label)
            print(f'loss: {loss}')
            #populates gradients of model parameters with respect to loss
            diff = loss.backward()
            #print(f'diff: {diff}')
            #opimisation step 
            optimiser.step()
            #the .grad associated with tensor .backward does not go back to 0 with every iteration (clearly this would cause issues with SGD), accordingly must re-zero the grad of the optimiser after each iteration. But don't do this before .step!
            optimiser.zero_grad()
            writer.add_scalar(name_writer, loss.item(), batch_idx)
            batch_idx += 1

    end_time = time()
    training_duration = end_time - start_time
    print(f'time for model training: {training_duration}')
    return training_duration, loss

def evaluate_model(model, val_dataloader, test_dataloader=None):

    model.eval()
    #we don't need the grad function here so switching it off with no_grad

    inference_latencies_list = []

    with torch.no_grad():
        for X, y in val_dataloader:
            #starting the timer for inference latency
            inference_latency_start = time()
            #making the prediction
            prediction = model(X)
            #appending the latency into a list
            inference_latencies_list.append(time() - inference_latency_start)
            val_loss = F.mse_loss(prediction, y)
            print(f'loss: {val_loss}')
            #need to instantiate r2 for this to work
            r2_inst = torchmetrics.R2Score()
            #need same dimensions, so flattenned prediction which was (x, 1) dimension
            prediction_flattened = torch.flatten(prediction)
            val_r2 = r2_inst(prediction_flattened, y)
            print(f'val_r2: {val_r2}')

    if test_dataloader == None:
        test_loss, test_r2 = 0, 0
           
    #we don't need the grad function here so switching it off with no_grad
    else: 
        with torch.no_grad():
            for X, y in test_dataloader:
                inference_latency_start = time()
                prediction = model(X)
                inference_latencies_list.append(time() - inference_latency_start)
                test_loss = F.mse_loss(prediction, y)
                print(f'loss: {test_loss}')
                #need to instantiate r2 for this to work
                r2_inst = torchmetrics.R2Score()
                #need same dimensions, so flattenned prediction which was (x, 1) dimension
                prediction_flattened = torch.flatten(prediction)
                test_r2 = r2_inst(prediction_flattened, y)
                print(f'test_r2: {test_r2}')

    mean_inference_latency = np.mean(inference_latencies_list)

    return val_loss, val_r2, test_loss, test_r2, mean_inference_latency


class CNN(torch.nn.Module):
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

def train_cnn(model, num_epochs, name_writer, dataloader):

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
            loss = F.cross_entropy(prediction, label)
            print(f'loss: {loss}')
            #populates gradients of model parameters with respect to loss
            diff = loss.backward()
            #print(f'diff: {diff}')
            #opimisation step 
            optimiser.step()
            #the .grad associated with tensor .backward does not go back to 0 with every iteration (clearly this would cause issues with SGD), accordingly must re-zero the grad of the optimiser after each iteration. But don't do this before .step!
            optimiser.zero_grad()
            writer.add_scalar(name_writer, loss.item(), batch_idx)
            batch_idx += 1

        end_time = time()
    training_duration = end_time - start_time

    return training_duration, loss

def save_nn_model(folder_path, model, hyperparameters, metrics)
    today = datetime.now()
    today = today.strftime('\%Y-%m-%d_%H%M%S')
    print(today)
    os_dir = str(Path(folder_path))
    print(os_dir)
    path_2 = str(os_dir + today)
    print(path_2)
    os.mkdir(path_2)
    print(os_dir)
    file_name = str(path_2) + '.pt'
    print(file_name)
    #saving the model
    torch.save(model.state_dict(), file_name)
    joblib.dump(model, f'{folder_path}/model.joblib')
    
    with open(f'{folder_path}/hyperparameters.json', 'a') as outfile:
        json.dump(hyperparameters, outfile)
    with open(f'{folder_path}/metrics.json', 'a') as outfile:
        json.dump(metrics, outfile)


if __name__ == '__main__':
    dataset = pd.read_csv(r'AirbnbDataSci\tabular_data\clean_tabular_data.csv')
    X_train, y_train, X_val, y_val, X_test, y_test =  split_dataset(dataset, ['Price_Night'], 42)
    
    batch_size = len(X_train)
    n_iters = 300
    num_epochs = n_iters / (len(X_train) / batch_size)
    num_epochs = int(num_epochs)
    dataloader_train = create_dataloader(X_train, y_train, batch_size)
    model_bl = NN()
    training_duration, train_loss = train_simple(model_bl, num_epochs, 'abc', dataloader_train)
    dataloader_val = create_dataloader(X_val, y_val, batch_size)
    val_loss, val_r2, test_loss, test_r2, mean_inference_latency = evaluate_model(model_bl, dataloader_val, test_dataloader=None)
    #do i make a function here?
    metrics = {training_duration, train_loss, val_loss, val_r2, test_loss, test_r2, mean_inference_latency}
    
    

    
    
 