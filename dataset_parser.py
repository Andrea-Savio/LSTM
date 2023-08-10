import json
#from operator import concat
import torch
import math
import numpy as np

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

    for id, detections in data['detections'].items():
        for detection in detections:    
            maxt = maxt + 1
        if maxt > max:
            max = maxt
        maxt = 0    
       
    print(max)    
        
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Array initialization, if size is not ok adjust it

    if (max % tentative_prediction_window != 0):
        max = max - (max % tentative_prediction_window) + tentative_prediction_window

    coordinates = np.zeros((counter, max, 5), dtype=float)
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
        index_depth = index_depth + 1
        index_rows = 0
    
    # Compute variation in position with respect to previous detection

    for i in range(counter):
        for j in range(1,max):
            coordinates[i, j, 4] = math.sqrt((coordinates[i, j, 0] - coordinates[i, j-1, 0])**2 + (coordinates[i, j, 1] - coordinates[i, j-1, 1])**2)

    #print(coordinates)
    return coordinates 

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
    data = dataset_parser('bytes-cafe-2019-02-07_0.json',35, True)    
    window_size = 35
    print(data)
    X_train = data[:,:(len(data)-window_size),:]
    Y_train = data[:,window_size:,:]
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    #X_train_sub = create_subsequence(X_train,window_size)
    #print(X_train_sub.shape)

    #print(X_train.shape)