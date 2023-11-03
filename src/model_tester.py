import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import utils.metrics as met
from pickle import load
from lstm_trainer3 import LSTM_Trainer
from clustering import dunn_index

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score                         
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from dataset_parser_multi_file import dataset_parser
from dataset_parser_multi_file import CustomDataset
from dataset_parser_multi_file import create_subsequence

#from torchviz import make_dot



if __name__ == '__main__':
    # Load what you need

    csv_file_path = 'ucy.csv'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LSTM_Trainer(2, 256, 3, 52)
    model.load_state_dict(torch.load("models/model_complete/3lstm_2d_complete.pt"))
    
    #x = torch.randn(1 ,35, 2)
    #y = model(x)
    #make_dot(y, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
    
    model.to(device)
    atc1 = torch.load('atc_20121031/atc_20121031_tensor1.pt')
    #atc1 = dataset_parser("bytes-cafe-2019-02-07_0.json",35,True)
    #atc1 = torch.tensor(atc1, dtype = torch.float32)
    scaler = load(open('scaler_2d_final.pkl', 'rb'))
    # Put the model in evaluation mode
    model.eval()
    eps = 2
    min_samples = 2
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Test input data (replace with your actual test data)
    #test_input = torch.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Perform the forward pass

    seq_length = 70
    fde = 0
    ade = 0
    maxdist = 0
    mr = 0
    iou = 0
    frequency = 7.5
    time_interval = 1/frequency

    inputs = atc1[:,:(atc1.size()[1] - seq_length),0:2]
    ground_truth = atc1[:,(seq_length):,0:2]

    points = inputs[:,0,0:2].cpu().detach().numpy()
    labels = dbscan.fit_predict(points)

    silhouette = silhouette_score(points, labels)
    davies_bouldin = davies_bouldin_score(points, labels)
    calinski_harabasz = calinski_harabasz_score(points, labels)
    dunn = dunn_index(points, labels)

    with torch.no_grad():
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(["Actual","Predicted"])
            num_samples, total, input_dim = inputs.size()
            num_subsequences = total - seq_length + 1
            for i in range(atc1.size()[0]):
                #i = i + 13 
                #print(i)
                for j in range(1, num_subsequences):
                    test_input_cont = np.zeros((1, seq_length, 8))
            
                    test_input_traj = inputs[i,j:j+seq_length,0:2]
            
                    #print(test_input_traj.shape)
                    test_input_cont[0,:,0] = atc1[i,j,2]
                    test_input_cont[0,:,2] = atc1[i,j,3]
                    test_input_cont[0,:,1] = math.sqrt((atc1[i, j, 0] - atc1[i, j-1, 0])**2 + (atc1[i, j, 1] - atc1[i, j-1, 1])**2)

                    test_input_cont[0, :, 3] = len(labels)
                    test_input_cont[0, :, 4] = silhouette
                    test_input_cont[0, :, 5]= davies_bouldin
                    test_input_cont[0, :, 6]= calinski_harabasz
                    test_input_cont[0, :, 7]= dunn

                    test_input_cont = torch.tensor(test_input_cont, dtype=torch.float32)
                    test_input_cont = test_input_cont.to(device)

                    #test_input_traj = torch.tensor(scaler.transform(test_input_traj.reshape(-1,2).reshape(test_input_traj.shape)), dtype = torch.float32)
                    test_input_traj = test_input_traj.view(1,seq_length,2)
                    #test_input_cont = atc1[i,:,2:]
                    test_input_traj = test_input_traj.to(device)
                    test_ground_truth = ground_truth[i,j:j+seq_length,0:2]
                    test_ground_truth = test_ground_truth.to(device)
                    test_ground_truth = test_ground_truth.view(1, seq_length, 2)
                    #print(test_input_traj)
                    test_output, cont = model(test_input_traj, test_input_cont)


                    test_output = test_output.cpu().detach().numpy()
                    #test_output = scaler.inverse_transform(test_output)
                    #test_output.to(device)

                    #test_output = test_output.cpu()
                    test_ground_truth = test_ground_truth.cpu().detach().numpy()

                    data = list(zip(test_ground_truth, test_output))
                    
                    writer.writerows(data)


                    #print(test_output)
                    #print(test_ground_truth)
                    #if j == num_subsequences - 1:
                    fde = fde + met.final_displacement_error(test_output[0], test_ground_truth[0])
                    ade = ade + met.average_displacement_error(test_output[0], test_ground_truth[0])
                    if maxdist < met.calculate_max_dist(test_output[0], test_ground_truth[0]):
                        maxdist = met.calculate_max_dist(test_output[0], test_ground_truth[0])
                    for k in range(seq_length):
                        iou = iou + met.calculate_iou([test_output[0,k,0] -1, test_output[0,k,1] - 1, test_output[0,k,0] + 1, test_output[0,k,1] + 1], [test_ground_truth[0,k,0] -1, test_ground_truth[0,k,1] - 1, test_ground_truth[0,k,0] + 1, test_ground_truth[0,k,1] + 1])
                    
                    if met.euclidean_distance(test_output[-1], test_ground_truth[-1]) < 10:
                        #print(met.euclidean_distance(test_output[-1], test_ground_truth[-1]))
                        mr = mr + 1
                #mr = mr/num_subsequences
                #ade = ade/(num_subsequences*seq_length)
                #fde = fde/num_subsequences
                #iou = iou/(total*seq_length)
            mr = mr/(num_subsequences*num_samples)
            ade = ade/(num_subsequences*seq_length*num_samples)
            fde = fde/(num_subsequences*num_samples)
            iou = iou/(total*seq_length*num_samples)
    # Post-process the output if necessary
    # Evaluate the model's predictions (e.g., calculate MSE or other metrics)
    # Display or save the results
    print("FDE: ", fde)
    print("ADE: ", ade)
    print("MaxDist: ", maxdist)
    print("IoU: ", iou)
    print("MR: ", mr)
