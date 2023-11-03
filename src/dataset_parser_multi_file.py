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
        data_path = 'train_dataset_with_activity/labels/labels_3d/'
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

    # Load the JSON data from the file
    if train:
        data_path = 'train_dataset_with_activity/labels/labels_3d/'
    else:
        data_path = 'test_dataset/test_detections/detections/detections_3d/'
    file_path = data_path + file_name

    with open(file_path) as file:
        data = json.load(file)

    for item, detections in data['labels'].items():
        for detection in detections:
            #print(int(detection['label_id'].split(":")[1]))
            #if int(detection['label_id'].split(":")[1]) == 36:
            #    counter = counter + 1
            if max < int(detection['label_id'].split(":")[1]):
                max = int(detection['label_id'].split(":")[1])
    
    counter = np.zeros(max)

    for item, detections in data['labels'].items():
        for detection in detections:
            counter[int(detection['label_id'].split(":")[1]) - 1] += 1        
            
    #print(max)
    #print(np.amax(counter))
        
    # Determine the dimensions of the array based on the data
    num_features = 11  # "cx," "cy," "cz", "ang"

    # Initialize the 3D NumPy array with zeros
    array_3d = np.zeros((max, int(np.amax(counter)), num_features))
    index = np.zeros(max + 1)

    # Populate the array with data from the JSON file
    for item, detections in data['labels'].items():
        for detection in detections:
            #print(item)
            
            
            label_id = int(detection['label_id'].split(":")[1])
            #print(label_id)
            
            
            #sequence_length = item['attributes']['num_points']
            box = detection['box']
            i = int(index[label_id])
            #print(i)
            
            #for i in range(sequence_length):
            array_3d[label_id - 1, i, 0] = box['cx']
            array_3d[label_id - 1, i, 1] = box['cy']
            array_3d[label_id - 1, i, 2] = box['cz']
            array_3d[label_id - 1, i, 3] = box['rot_z']

            index[label_id] += 1
            #print(array_3d[13,0:50])
                
    #print(index)

    todelete = []
    for i in range(len(index)):
        if index[i] == 0:
            todelete.append(i-1)

    array_3d = np.delete(array_3d,todelete,axis=0)

    # Compute variation in position with respect to previous detection

    for i in range(max - len(todelete)):
        for j in range(1,int(np.amax(counter))):
            array_3d[i, j, 4] = math.sqrt((array_3d[i, j, 0] - array_3d[i, j-1, 0])**2 + (array_3d[i, j, 1] - array_3d[i, j-1, 1])**2)

    
    # Compute speed (assumption is constant speed throughout the motion)

    frequency = 7.5
    time_interval = 1/frequency

    for i in range(max - len(todelete)):
        for j in range(int(np.amax(counter))):
            if (array_3d[i,j,4] != 0) or (j == 0):
                array_3d[i,j,5] = (array_3d[i,1,4] - array_3d[i,0,4])/time_interval

    #return coordinates, single_lengths
    
    # Clustering
    
    eps = 2
    min_samples = 2
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    points = array_3d[:,0,0:2]
    #print(points)
    labels = dbscan.fit_predict(points)
    #print(len(labels))
    #print(labels)

    

    for i in range(max - len(todelete)):
        #x = coordinates[i,1,0]
        #y = coordinates[i,1,1]
        #new = np.array([x,y])
        #data.reshape(-1,1)
        #label = dbscan.fit_predict([new])[0]
        #print(label)
        array_3d[i,:,6] = len(labels)

    silhouette = silhouette_score(points, labels)
    davies_bouldin = davies_bouldin_score(points, labels)
    calinski_harabasz = calinski_harabasz_score(points, labels)
    dunn = dunn_index(points, labels)

    array_3d[:, :, 7] = silhouette
    array_3d[:, :, 8] = davies_bouldin
    array_3d[:, :, 9] = calinski_harabasz
    array_3d[:, :, 10] = dunn
    

    #print(coordinates
    
    #return array_3d        

    #array_3d = np.delete(array_3d,todelete,axis=0)
    #index = np.delete(index,todelete)
    #print(index)
    #print(array_3d.shape)
    return array_3d

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
    """
    train_files = ['packard-poster-session-2019-03-20_1.json', 'serra-street-2019-01-30_0.json', 'bytes-cafe-2019-02-07_0.json', 'nvidia-aud-2019-01-25_0.json', 'discovery-walk-2019-02-28_0.json', 'memorial-court-2019-03-16_0.json', 'gates-foyer-2019-01-17_0.json', 'tressider-2019-03-16_2.json', 'huang-basement-2019-01-25_0.json', 'gates-basement-elevators-2019-01-17_1.json', 'discovery-walk-2019-02-28_1.json', 'tressider-2019-04-26_2.json', 'svl-meeting-gates-2-2019-04-08_0.json', 'meyer-green-2019-03-16_1.json', 'gates-to-clark-2019-02-28_1.json', 'svl-meeting-gates-2-2019-04-08_1.json', 'gates-ai-lab-2019-02-08_0.json', 'tressider-2019-03-16_0.json', 'stlc-111-2019-04-19_0.json', 'tressider-2019-04-26_0.json', 'gates-to-clark-2019-02-28_0.json', 'gates-ai-lab-2019-04-17_0.json', 'huang-2-2019-01-25_0.json', 'tressider-2019-04-26_1.json', 'stlc-111-2019-04-19_1.json', 'lomita-serra-intersection-2019-01-30_0.json', 'hewlett-class-2019-01-23_1.json', 'cubberly-auditorium-2019-04-22_1.json', 'hewlett-packard-intersection-2019-01-24_0.json', 'tressider-2019-03-16_1.json', 'clark-center-2019-02-28_1.json', 'huang-lane-2019-02-12_0.json', 'tressider-2019-04-26_3.json', 'nvidia-aud-2019-04-18_1.json', 'huang-intersection-2019-01-22_0.json', 'packard-poster-session-2019-03-20_2.json', 'food-trucks-2019-02-12_0.json', 'packard-poster-session-2019-03-20_0.json', 'outdoor-coupa-cafe-2019-02-06_0.json', 'forbes-cafe-2019-01-22_0.json', 'nvidia-aud-2019-04-18_0.json', 'meyer-green-2019-03-16_0.json', 'quarry-road-2019-02-28_0.json', 'cubberly-auditorium-2019-04-22_0.json', 'nvidia-aud-2019-04-18_2.json', 'hewlett-class-2019-01-23_0.json', 'jordan-hall-2019-04-22_0.json', 'indoor-coupa-cafe-2019-02-06_0.json', 'clark-center-intersection-2019-02-28_0.json', 'huang-2-2019-01-25_1.json', 'stlc-111-2019-04-19_2.json', 'gates-159-group-meeting-2019-04-03_0.json', 'gates-basement-elevators-2019-01-17_0.json', 'clark-center-2019-02-28_0.json']
    #for file in train_files:
    data = dataset_parser("bytes-cafe-2019-02-07_0.json",25,True)
    #print(data[25,50:200])
    #    break
    """



    
    total =  torch.tensor([], dtype=torch.float32)
    train_files = ['packard-poster-session-2019-03-20_1.json', 'bytes-cafe-2019-02-07_0.json', 'memorial-court-2019-03-16_0.json', 'huang-basement-2019-01-25_0.json', 'gates-basement-elevators-2019-01-17_1.json', 'tressider-2019-04-26_2.json', 'svl-meeting-gates-2-2019-04-08_0.json', 'gates-to-clark-2019-02-28_1.json', 'svl-meeting-gates-2-2019-04-08_1.json', 'gates-ai-lab-2019-02-08_0.json', 'tressider-2019-03-16_0.json', 'stlc-111-2019-04-19_0.json', 'huang-2-2019-01-25_0.json', 'hewlett-packard-intersection-2019-01-24_0.json', 'tressider-2019-03-16_1.json', 'clark-center-2019-02-28_1.json', 'huang-lane-2019-02-12_0.json', 'packard-poster-session-2019-03-20_2.json', 'packard-poster-session-2019-03-20_0.json', 'forbes-cafe-2019-01-22_0.json', 'nvidia-aud-2019-04-18_0.json', 'meyer-green-2019-03-16_0.json', 'cubberly-auditorium-2019-04-22_0.json', 'jordan-hall-2019-04-22_0.json', 'clark-center-intersection-2019-02-28_0.json', 'gates-159-group-meeting-2019-04-03_0.json', 'clark-center-2019-02-28_0.json']

    for file in train_files:
        data = dataset_parser(file,35,True)
        #print(data)
        
        data = data.reshape(-1)
        data = torch.tensor(data,dtype=torch.float32)    
        total = torch.cat([total, data], dim=0)

    #print(total)
    num_samples = total.shape[0]
    num_features = torch.prod(torch.tensor(total.shape[1:])).item()
    reshaped_tensor = total.reshape(-1,2)#.view(-1, 1)
    #print(reshaped_tensor)
    # Convert the PyTorch tensor to a NumPy array
    numpy_array = reshaped_tensor.numpy()

    # Apply StandardScaler
    scaler = StandardScaler()
    scaled_numpy_array = scaler.fit_transform(numpy_array)
    #print(scaled_numpy_array)

    #dump(scaler, open('scaler_2d_final.pkl', 'wb'))
    #print("Scaler saved")


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