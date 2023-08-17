import json
import os
#from operator import concat
import torch
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score                         
from sklearn.metrics import pairwise_distances
from clustering import dunn_index

class CustomDataset(Dataset):

    def __init__(self, data, seq_length, transform=None, target_transform=None):
        self.seq_length = seq_length
        self.X_train = data[:,:(data.shape[1]-seq_length),:]
        self.Y_train = data[:,(seq_length):,:]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X_train)
    
    def __getitem__(self, idx):
        sample = self.X_train[idx,:,:]
        label = self.Y_train[idx,:,:]
        #mask_xtrain = self.X_train != 0
       # mask_ytrain = self.Y_train != 0
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label #mask_xtrain, mask_ytrain    


# This function parses the datasets and creates a tensor with the coordinates for each timestep for each person tracked in the dataset.

def dataset_parser(file_name, pred_window, train: bool):

# Define file path and some useful variables for array dimension

    if train:
        data_path = 'train_dataset_with_activity/detections/detections_3d/'
    else:
        data_path = 'test_dataset/test_detections/detections/detections_3d/'
    file_path = data_path + file_name

    with open(file_path) as file:
        data = json.load(file)

    counter = 0
    index_rows = 0
    index_depth = 0
    idis = []
    max = 0
    maxt = 0
    first = True
    tentative_prediction_window = pred_window

    #-----------------------------------------------------------------------------------------------------------------------------------

    # Count number of people and number of detection for each person for later

    for id, detections in data['detections'].items():
        #print("ID: {id}")
        #print(id)
        idis.append(id)

    #print(len(idis))
    counter = len(idis)
    del(idis)   

    single_lengths = []
    for id, detections in data['detections'].items():
        for detection in detections:    
            maxt = maxt + 1
        single_lengths.append(maxt)    
        if maxt > max:
            max = maxt
        maxt = 0    
       
    print(max)    
        
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Array initialization, if size is not ok adjust it

    if (max % tentative_prediction_window != 0):
        max = max - (max % tentative_prediction_window) + tentative_prediction_window

    coordinates = np.zeros((counter, max, 11), dtype=float)
    #coordinates2 = np.zeros((max, 3, counter), dtype=float)

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Extract cx, cy, cz, rotation for each different detection and for each different id and add to array
    for id, detections in data['detections'].items():
        
        #print(detections)
    
        for detection in detections:
            cx = detection['box']['cx']
            cy = detection['box']['cy']
            cz = detection['box']['cz']
            rot = detection['box']['rot_z']
                
            coordinates[index_depth, index_rows, 0] = cx
            coordinates[index_depth, index_rows, 1] = cy
            coordinates[index_depth, index_rows, 2] = cz
            coordinates[index_depth, index_rows, 3] = rot
            index_rows = index_rows + 1
        if index_rows < max:
            coordinates[index_depth, index_rows:max, 0] = coordinates[index_depth, index_rows - 1, 0]
            coordinates[index_depth, index_rows:max, 1] = coordinates[index_depth, index_rows - 1, 1]
            coordinates[index_depth, index_rows:max, 2] = coordinates[index_depth, index_rows - 1, 2]
            coordinates[index_depth, index_rows:max, 3] = coordinates[index_depth, index_rows - 1, 3]

        index_depth = index_depth + 1
        index_rows = 0
    
    # Compute variation in position with respect to previous detection

    for i in range(counter):
        for j in range(1,max):
            coordinates[i, j, 4] = math.sqrt((coordinates[i, j, 0] - coordinates[i, j-1, 0])**2 + (coordinates[i, j, 1] - coordinates[i, j-1, 1])**2)

    
    # Compute speed (assumption is constant speed throughout the motion)

    frequency = 7.5
    time_interval = 1/frequency

    for i in range(counter):
        for j in range(max):
            if (coordinates[i,j,4] != 0) or (j == 0):
                coordinates[i,j,5] = (coordinates[i,1,4] - coordinates[i,0,4])/time_interval

    #return coordinates, single_lengths
    
    # Clustering
    
    eps = 0.5
    min_samples = 2
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    points = coordinates[:,max-1,0:2]
    #print(points)
    labels = dbscan.fit_predict(points)
    #print(len(labels))
    #print(labels)

    

    for i in range(counter):
        #x = coordinates[i,1,0]
        #y = coordinates[i,1,1]
        #new = np.array([x,y])
        #data.reshape(-1,1)
        #label = dbscan.fit_predict([new])[0]
        #print(label)
        coordinates[i,:,6] = labels[i]

    silhouette = silhouette_score(points, labels)
    davies_bouldin = davies_bouldin_score(points, labels)
    calinski_harabasz = calinski_harabasz_score(points, labels)
    dunn = dunn_index(points, labels)

    coordinates[:, :, 7] = silhouette
    coordinates[:, :, 8] = davies_bouldin
    coordinates[:, :, 9] = calinski_harabasz
    coordinates[:, :, 10] = dunn
    

    #print(coordinates
    return coordinates, single_lengths 
    """
    """
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create subsequences of the desired time window (7.5 Hz)

def create_subsequence(data, sequence_length):
    num_samples, seq_length, input_dim = data.size()
    num_subsequences = seq_length - sequence_length + 1
    subsequences = []
    for i in range(num_subsequences):
        subsequence = data[:,i:i+sequence_length,:]
        subsequences.append(subsequence)

    return torch.stack(subsequences)               
    

if __name__ == '__main__':
    data, l = dataset_parser('bytes-cafe-2019-02-07_0.json',35, True)    
    window_size = 35
    #print(data[0,0,0,0,0,0,:])
    X_train = data[:,:(len(data)-window_size),:]
    Y_train = data[:,window_size:,:]
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    #X_train_sub = create_subsequence(X_train,window_size)
    #print(X_train_sub.shape)

    print(X_train)