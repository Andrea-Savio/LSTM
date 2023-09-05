#!/usr/bin/env python3

import rospy
import time
from lstm_trainer3 import LSTM_Trainer
import torch
import torch.nn
import numpy as np
from geometry_msgs.msg import Pose
from spencer_tracking_msgs.msg import TrackedPersons
from spencer_tracking_msgs.msg import TrackedPerson
from trajectory_prediction.msg import PredictedTrajectories
from trajectory_prediction.msg import PredictedTrajectory
from dataset_parser import dataset_parser
from pickle import load

if __name__=="__main__":

    scaler = load(open('scaler_full.pkl', 'rb'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LSTM_Trainer(3, 256, 3, 35)
    model.load_state_dict(torch.load("models/model_complete/3layers_10epochs.pt"))
    #model.load_state_dict(torch.load("models/model7/model7_batch_32_final.pt"))
    #model.to(device)

    model.eval()

    data, l = dataset_parser("bytes-cafe-2019-02-07_0.json",35,True)

    X_train = data[1,:35,0:3]
    X_train2 = data[1,:35,3:]
    X_train = torch.tensor(scaler.transform(X_train.reshape(-1,1)).reshape(X_train.shape), dtype=torch.float32)
    X_train2 = torch.tensor(X_train2, dtype=torch.float32)
    X_train = X_train.view(1,35,3)
    X_train2 = X_train2.view(1,35,8)
    print(X_train)
    out1,out2 = model(X_train,X_train2)
    out1 = out1.cpu().detach().numpy()
    out1 = scaler.inverse_transform(out1)
    Y_train = torch.tensor(data[1,35:70,0:3],dtype=torch.float32)
    print(Y_train)
    print(out1)

