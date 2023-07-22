import json
#from operator import concat
import torch
import numpy as np

# This function parses the datasets and creates a tensor with the coordinates for each timestep for each person tracked in the dataset.

def dataset_parser_multi(pred_window):

    # Define file path and some useful variables for array dimension

    counter = 0
    index_rows = 0
    index_depth = 0
    idis = []
    max = 0
    maxt = 0
    first = True
    tentative_prediction_window = pred_window

    data_path = 'train_dataset_with_activity/detections/detections_3d/'
    json_files = ["bytes-cafe-2019-02-07_0.json"]#,"clark-center-2019-02-28_0.json", "clark-center-2019-02-28_1.json"]
    #file_path = data_path + file_name

    data = []

    #-----------------------------------------------------------------------------------------------------------------------------------

    # Count number of people and number of detection for each person for each dataset file, create a complete data object

    for file in json_files:
        file_path = data_path + file
        with open(file_path) as file:

            loaded_data = json.load(file)

            for id, detections in loaded_data['detections'].items():
                counter = counter + 1
                for detection in detections:    
                    maxt = maxt + 1
                if maxt > max:
                    max = maxt
                maxt = 0    

            coordinates = np.zeros((max, 3, counter), dtype=float)

            for detection in detections:
                cx = detection['box']['cx']
                cy = detection['box']['cy']
                cz = detection['box']['cz']
            
                coordinates[index_rows, 0, index_depth] = cx
                coordinates[index_rows, 1, index_depth] = cy
                coordinates[index_rows, 2, index_depth] = cz
                index_rows = index_rows + 1
            index_depth = index_depth + 1
            index_rows = 0

            #data.append(loaded_data)

    #-----------------------------------------------------------------------------------------------------------------------------------
    """
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
       
    #print(max)    
    """    
    #---------------------------------------------------------------------------------------------------------------------------------

    # Array initialization, if size is not ok adjust it

    if (max % tentative_prediction_window != 0):
        max = max - (max % tentative_prediction_window) + tentative_prediction_window

    coordinates = np.zeros((max, 3, counter), dtype=float)
    #coordinates2 = np.zeros((max, 3, counter), dtype=float)

    #---------------------------------------------------------------------------------------------------------------------------------

    # Extract cx, cy, cz for each different detection and for each different id and add to array
    for id, detections in data['detections'].items():
        
        #print(detections)
    
        for detection in detections:
            cx = detection['box']['cx']
            cy = detection['box']['cy']
            cz = detection['box']['cz']
            
            coordinates[index_rows, 0, index_depth] = cx
            coordinates[index_rows, 1, index_depth] = cy
            coordinates[index_rows, 2, index_depth] = cz
            index_rows = index_rows + 1
        index_depth = index_depth + 1
        index_rows = 0

    #print(coordinates)
    return coordinates        
    

if __name__ == '__main__':
    data = dataset_parser_multi(35)    
    window_size = 35

    print(len(data))
    print(len(data[0]))
    print(len(data[0][0]))

    """
    X_train = data[:(len(data)-window_size),:,:]
    Y_train = data[window_size:,:,:]
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    print(X_train.size())
    """