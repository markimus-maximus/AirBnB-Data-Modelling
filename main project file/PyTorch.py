import numpy as np
import pandas as pd
import tabular_data as td
import torch
import torch.nn.functional as F
from IPython.display import display
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split
from torch import tensor
from pathlib import Path

#prepare data first: 

# read the scv

data = pd.read_csv(r'C:\Users\marko\DS Projects\AirBnB-Data-Modelling\main project file\AirbnbDataSci\tabular_data\clean_tabular_data.csv')
#drop non-numerical
features_dropped = data.drop(['Category', 'ID', 'Title','Description','Amenities','Location','url'], axis=1)
#index only the label and covert to tensor of floats
label_all = tensor(features_dropped['Price_Night']).float()
#index features by dropping label and covert to tensor of floats
features_all = tensor(features_dropped.drop('Price_Night', axis=1).values).float()  

#class to make the data iterable by index, reurning tuple of tensor features, tensor label
class AirbnbNightlyPriceImageDataset(Dataset):
    
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


#generating a datset object according to the class above
dataset = AirbnbNightlyPriceImageDataset(features_all, label_all)

#splitting the data into train, val, test subsets with random_split
train_dataset, test_val_dataset = random_split(dataset, [0.7, 0.3])
val_dataset, test_dataset =  random_split(test_val_dataset, [0.6, 0.4])

print(f'len(train_dataset):{ len(train_dataset)}, len(val_dataset): {len(val_dataset)}, len(test_dataset): {len(test_dataset)}')

#declare batch size variable (the number of examples used to train the model each time)
batch_size = 10

#num_epochs = n_iters / (len(features) / batch_size)
# num_epochs = int(num_epochs)

num_epochs = 20
dataloader_train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

class LinearRegression(torch.nn.Module):
    
    #initialise the parameters
    def __init__(self):
        super().__init__()
        #for linear problem, give a weighted combination of features plus a bias. Takes the number of inputs and outputs required
        self.linear_layer = torch.nn.Linear(11, batch_size)

    #describes behaviour when the class is called, similar to __call__
    def forward(self, features):
        #use the layers to process the features, returning a prediction
        return self.linear_layer(features)

model = LinearRegression()
        




def train(model, num_epochs):
    
    for epoch in range(num_epochs):
        #iterate through batches for the number of epochs declared
        for batch in dataloader_train:
            #print(batch)
            #unpack the batch into features and labels
            features, label = batch
            #make a prediction from the model based on a batch of features
            prediction = model(features)
            print(prediction)
            #calculate the loss- used to improve with gradient descent
            loss = F.mse_loss(prediction, label)
            print(f'loss: {loss}')
            #compute gradients with respect to loss and model parameters
            diff = loss.backward()
            print(f'diff: {diff}')
            break
            #opimisation step



train(model, num_epochs=num_epochs)



    

    
