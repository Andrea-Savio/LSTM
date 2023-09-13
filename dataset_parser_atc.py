import json
import os
#from operator import concat
import torch
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from pickle import dump
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score                         
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from clustering import dunn_index

import csv

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

    counter = []
    people = []
    i = 0
    #max = 0

    with open('atc-20121024.csv', 'r') as csvfile:
        # Read the CSV file
        csvreader = csv.reader(csvfile)

        # Choose the column you want to extract by index or name
        id_column = 1  # Replace with the desired column index
        x_column = 2
        y_column = 3
        z_column = 4
        vel_column = 5
        ang_column = 6

        column_indices = [1,2,3,4,5,6]
        # OR
        # column_name = "ColumnName"  # Replace with the desired column name
        # header = next(csvreader)  # Read the header row
        # column_index = header.index(column_name)  # Find the index of the column by name

        # Extract the selected column

        selected_columns = [[] for _ in column_indices]
        
        selected_column_id = [row[1] for row in csvreader]
        """
        selected_column_x = [row[x_column] for row in csvreader]
        selected_column_y = [row[y_column] for row in csvreader]
        selected_column_z = [row[z_column] for row in csvreader]
        selected_column_vel = [row[vel_column] for row in csvreader]
        selected_column_ang = [row[ang_column] for row in csvreader]
        """

        # Use the selected column (e.g., print its values)
        for value in selected_column_id:
            i = i + 1
            if value not in people:
                people.append(value)
                counter.append(1)
                #counter = counter + 1
                #print(value)
            else:
                up = people.index(value)
                counter[up] = counter[up] + 1   
            if i == 7:
                break

        #print(len(people))
        #print(counter)
        #print()

        coordinates = np.zeros((len(people), max(counter), 4), dtype=float)
        for n in range(len(counter)):
            counter[n] = 0

    k = 0
    with open('atc-20121024.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if k == 7:
                break
            k = k + 1
            #print("in")
            #print(row)
            j = people.index(row[1])
            coordinates[j,counter[j],0] = float(row[2])/1000
            coordinates[j,counter[j],1] = float(row[3])/1000
            #coordinates[j,counter[j],2] = float(row[4])/1000
            coordinates[j,counter[j],2] = float(row[5])/1000
            coordinates[j,counter[j],3] = (row[6])
            counter[j] = counter[j] + 1
            
    return coordinates
    """
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
       
    #print(max)    
        
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Array initialization, if size is not ok adjust it

    if (max % tentative_prediction_window != 0):
        max = max - (max % tentative_prediction_window) + tentative_prediction_window

    print(max) 

    coordinates = np.zeros((counter, max, 11), dtype=float)
    #coordinates = np.zeros((counter, 210, 3), dtype=float)

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
    
    data = dataset_parser('asd', 35, True)
    #print(data)
    x = torch.tensor(data, dtype=torch.float32)
    torch.save(x, 'atc_20121024_tensor.pt')
    print("Dataset saved")

    #xx = torch.load('tensor.pt')
    #print(xx)
    
    """
    total =  torch.tensor([], dtype=torch.float32)
    train_files = ['packard-poster-session-2019-03-20_1.json', 'serra-street-2019-01-30_0.json', 'bytes-cafe-2019-02-07_0.json', 'nvidia-aud-2019-01-25_0.json', 'discovery-walk-2019-02-28_0.json', 'memorial-court-2019-03-16_0.json', 'gates-foyer-2019-01-17_0.json', 'tressider-2019-03-16_2.json', 'huang-basement-2019-01-25_0.json', 'gates-basement-elevators-2019-01-17_1.json', 'discovery-walk-2019-02-28_1.json', 'tressider-2019-04-26_2.json', 'svl-meeting-gates-2-2019-04-08_0.json', 'meyer-green-2019-03-16_1.json', 'gates-to-clark-2019-02-28_1.json', 'svl-meeting-gates-2-2019-04-08_1.json', 'gates-ai-lab-2019-02-08_0.json', 'tressider-2019-03-16_0.json', 'stlc-111-2019-04-19_0.json', 'tressider-2019-04-26_0.json', 'gates-to-clark-2019-02-28_0.json', 'gates-ai-lab-2019-04-17_0.json', 'huang-2-2019-01-25_0.json', 'tressider-2019-04-26_1.json', 'stlc-111-2019-04-19_1.json', 'lomita-serra-intersection-2019-01-30_0.json', 'hewlett-class-2019-01-23_1.json', 'cubberly-auditorium-2019-04-22_1.json', 'hewlett-packard-intersection-2019-01-24_0.json', 'tressider-2019-03-16_1.json', 'clark-center-2019-02-28_1.json', 'huang-lane-2019-02-12_0.json', 'tressider-2019-04-26_3.json', 'nvidia-aud-2019-04-18_1.json', 'huang-intersection-2019-01-22_0.json', 'packard-poster-session-2019-03-20_2.json', 'food-trucks-2019-02-12_0.json', 'packard-poster-session-2019-03-20_0.json', 'outdoor-coupa-cafe-2019-02-06_0.json', 'forbes-cafe-2019-01-22_0.json', 'nvidia-aud-2019-04-18_0.json', 'meyer-green-2019-03-16_0.json', 'quarry-road-2019-02-28_0.json', 'cubberly-auditorium-2019-04-22_0.json', 'nvidia-aud-2019-04-18_2.json', 'hewlett-class-2019-01-23_0.json', 'jordan-hall-2019-04-22_0.json', 'indoor-coupa-cafe-2019-02-06_0.json', 'clark-center-intersection-2019-02-28_0.json', 'huang-2-2019-01-25_1.json', 'stlc-111-2019-04-19_2.json', 'gates-159-group-meeting-2019-04-03_0.json', 'gates-basement-elevators-2019-01-17_0.json', 'clark-center-2019-02-28_0.json']
    for file in train_files:
        data, l = dataset_parser(file,35,True)
        data = torch.tensor(data,dtype=torch.float32)    
        total = torch.cat([total, data], dim=0)

    num_samples = total.shape[0]
    num_features = torch.prod(torch.tensor(total.shape[1:])).item()
    reshaped_tensor = total.view(-1, 1)
    print(reshaped_tensor.shape)
    # Convert the PyTorch tensor to a NumPy array
    numpy_array = reshaped_tensor.numpy()

    # Apply StandardScaler
    scaler = StandardScaler()
    scaled_numpy_array = scaler.fit_transform(numpy_array)

    #dump(scaler, open('scaler_full.pkl', 'wb'))
    #print("Scaler saved")
    """

    """    
    #print(data[0,0,0,0,0,0,:])
    X_train = data[:,:(len(data)-window_size),:]
    Y_train = data[:,window_size:,:]
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    #X_train_sub = create_subsequence(X_train,window_size)
    #print(X_train_sub.shape)

    print(X_train)
    """