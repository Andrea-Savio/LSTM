import torch
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable 
import numpy as np
from pickle import dump, load

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score                         
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#from matplotlib import pyplot as plt

from dataset_parser_multi_file import dataset_parser
from dataset_parser_multi_file import CustomDataset
from dataset_parser_multi_file import create_subsequence

class LSTM_Trainer(nn.Module):

    # Model definition

    def __init__(self, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_Trainer, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.output_size = input_size

        self.lstm_xyz = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.2)
        #self.bn = nn.BatchNorm1d(hidden_size)
        #self.linear_context = nn.Linear(256,128)
    
        self.fc1 = nn.Linear(self.hidden_size, 128)
        #self.relu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 64)
        #self.relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(64,2)
        #self.relu3 = nn.LeakyReLU(0.2)
        #self.dropout = nn.Dropout(p=0.2)
        #self.fc3 = nn.Linear(128,64)
        #self.fc4 = nn.Linear(64,3)

        #self.lstm_xyz1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        #self.lstm_context1 = nn.LSTM(input_size=2, hidden_size = hidden_size, num_layers = num_layers)

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Define how data flows into the network
    
    def forward(self, xyz):
        h_0xyz = Variable(torch.zeros(self.num_layers, xyz.size(1), self.hidden_size, device=xyz.device)) #hidden state
        c_0xyz = Variable(torch.zeros(self.num_layers, xyz.size(1), self.hidden_size, device=xyz.device)) #internal state

        #h_0cont = Variable(torch.zeros(self.num_layers, context.size(1), self.hidden_size, device=context.device)) 
        #c_0cont = Variable(torch.zeros(self.num_layers, context.size(1), self.hidden_size, device=context.device))
        
        # Propagate input through LSTM
        
        out_xyz, (hn_xyz, cn_xyz) = self.lstm_xyz(xyz, (h_0xyz, c_0xyz)) #lstm with input, hidden, and internal state
        #out_xyz, _ = rnn_utils.pad_packed_sequence(packed_out_xyz, batch_first=True)

        #out_xyz = out_xyz.transpose(1, 2)  # Transpose to [batch_size, hidden_size, sequence_length] for BatchNorm
        #out_xyz = self.bn(out_xyz)
        #out_xyz = out_xyz.transpose(1, 2)

        #out_context = self.linear_context(context)
        #print(out_xyz)
        #print(out_context)
        #combined = torch.cat((out_xyz, out_context),dim=2)
        #print(combined)
        
        out = self.fc1(out_xyz)
        #out = self.relu1(out)
        out = self.fc2(out)
        #out = self.relu2(out)
        output_xyz = self.fc3(out)
        """
        if (xyz[0, self.seq_length - 1, 0] > 0 and output_xyz[0,0,0] < 0):
            print("in")
            output_xyz[0,:,0] = -output_xyz[0,:,0]
        if (xyz[0,self.seq_length - 1,1] > 0 and output_xyz[0,0,1] < 0):
            output_xyz[0,:,1] = -output_xyz[0,:,1]
        if (xyz[0, self.seq_length - 1, 0] < 0 and output_xyz[0,0,0] > 0):
            print("in")
            output_xyz[0,:,0] = -output_xyz[0,:,0]
        if (xyz[0,self.seq_length - 1,1] < 0 and output_xyz[0,0,1] > 0):
            output_xyz[0,:,1] = -output_xyz[0,:,1]
        """
        #output_xyz = self.relu3(output_xyz)
        #output_xyz = self.fc4(out)
        #output_xyz = self.linear(out_xyz)

        #output_xyz = self.lstm_xyz1(out[:,0:3])
        #output_cont = self.lstm_context1(out[:,3:])
        #out = self.relu(hn)
        
        #hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        #out = self.relu(hn)
        #out = self.fc_1(out) #first Dense
        #out = self.relu(out) #relu
        #out = self.fc(out) #Final Output
        
        return output_xyz #, output_cont
    """

    def forward(self, x, future_steps):
        # x has shape (batch_size, sequence_length, input_size)
        predictions = []
        lstm_out, _ = self.lstm_xyz(x)
        
        # Initialize the first prediction with zeros
        prediction = torch.zeros(x.size(0), 1, x.size(2)).to(x.device)
        
        # Generate autoregressive predictions
        for _ in range(future_steps):
            lstm_out, _ = self.lstm_xyz(torch.cat([x, prediction], dim=1))
            prediction = self.fc1(lstm_out[:, -1:])
            predictions.append(prediction)
        
        return torch.cat(predictions, dim=1)
    """
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Save model
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

	#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	# Load model

    def load_model(self, path):
        self = torch.load(path)
	
	#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------      

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_files = ['packard-poster-session-2019-03-20_1.json', 'bytes-cafe-2019-02-07_0.json', 'memorial-court-2019-03-16_0.json', 'huang-basement-2019-01-25_0.json', 'gates-basement-elevators-2019-01-17_1.json', 'tressider-2019-04-26_2.json', 'svl-meeting-gates-2-2019-04-08_0.json', 'gates-to-clark-2019-02-28_1.json', 'svl-meeting-gates-2-2019-04-08_1.json', 'gates-ai-lab-2019-02-08_0.json', 'tressider-2019-03-16_0.json', 'stlc-111-2019-04-19_0.json', 'huang-2-2019-01-25_0.json', 'hewlett-packard-intersection-2019-01-24_0.json', 'tressider-2019-03-16_1.json', 'clark-center-2019-02-28_1.json', 'huang-lane-2019-02-12_0.json', 'packard-poster-session-2019-03-20_2.json', 'packard-poster-session-2019-03-20_0.json', 'forbes-cafe-2019-01-22_0.json', 'nvidia-aud-2019-04-18_0.json', 'meyer-green-2019-03-16_0.json', 'cubberly-auditorium-2019-04-22_0.json', 'jordan-hall-2019-04-22_0.json', 'clark-center-intersection-2019-02-28_0.json', 'gates-159-group-meeting-2019-04-03_0.json', 'clark-center-2019-02-28_0.json']
    test_files = ['packard-poster-session-2019-03-20_1.json', 'memorial-court-2019-03-16_0.json', 'bytes-cafe-2019-02-07_0.json']

    train_files_temp = ['packard-poster-session-2019-03-20_1.json']
    input_dim = 2
    num_layers = 3
    seq_length =  35 #35
    hidden_size = 128
    num_epochs = 10
    eps = 1e-6
    #scaled = False

    model = LSTM_Trainer(input_dim, hidden_size, num_layers, seq_length)
    #model.load_state_dict(torch.load("models/model_traj/3lstm_3h128_2d_20_unscaled.pt"))
    model = model.to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    #scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = load(open('scaler_2d_final.pkl', 'rb'))
    
    #---------------------------------------------------------------------------------------------------------------------------------

    # Training loop
    
    print("Begin training phase.")
    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch))
        for file in train_files_temp:

            with torch.no_grad():
                print("Data file: " + file)

                data = dataset_parser(file, seq_length, True)

                data = torch.load('atc_tracking_part1/atc_20121024/atc_20121024_tensor1.pt')

                #packed_sequences = rnn_utils.pack_padded_sequence(data,
                                                    #lengths, batch_first=True, enforce_sorted=False)
                dataset = CustomDataset(data, seq_length)

                train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

                #X_train, Y_train = next(iter(train_dataloader))

                #X_train = data[:,:(data.shape[1]-seq_length),:]
                #print(X_train)
                #Y_train = data[:,(seq_length):,:]

                #mask_xtrain = X_train != 0
                #mask_ytrain = Y_train != 0
        
                #X_train = torch.tensor(X_train, dtype=torch.float32)
                #X_train = X_train.unsqueeze(dim=0)
                #Y_train = torch.tensor(Y_train, dtype=torch.float32)
                #Y_train = Y_train.unsqueeze(dim=0)
                
                #X_train = X_train.to(device)
                #Y_train = Y_train.to(device)
                # Check corresponding sizes
                #print(X_train.size())
                #print(Y_train.size())

                print("Data ready, start training.")
                
                #num_samples, total, input_dim = X_train.size()
                #num_subsequences = total - seq_length + 1
            #for epoch in range(num_epochs):
                #print("Epoch: " + str(epoch))
            for X_train, Y_train in train_dataloader:
                #X_train = rnn_utils.pack_padded_sequence(X_train,
                #                                  lengths[0:63], batch_first=True, enforce_sorted=False)
                #Y_train = rnn_utils.pack_padded_sequence(Y_train,
                #                                  lengths[0:63], batch_first=True, enforce_sorted=False)

                  
                #if not scaled:
                #     X_train[:,:,0:3] = torch.tensor(scaler.fit_transform(X_train[:,:,0:3].reshape(-1,1)).reshape(X_train[:,:,0:3].shape), dtype=torch.float32)
                #    Y_train[:,:,0:3] = torch.tensor(scaler.transform(Y_train[:,:,0:3].reshape(-1,1)).reshape(Y_train[:,:,0:3].shape), dtype=torch.float32)
                #    scaled = True
                #print(X_train)
                #X_train[:,:,0:2] = torch.tensor(scaler.transform(X_train[:,:,0:2].reshape(-1,2)).reshape(X_train[:,:,0:2].shape), dtype=torch.float32)
                #Y_train[:,:,0:2] = torch.tensor(scaler.transform(Y_train[:,:,0:2].reshape(-1,2)).reshape(Y_train[:,:,0:2].shape), dtype=torch.float32)
                #print(X_train)
                exit = 0
                num_samples, total, input_dim = X_train.size()
                num_subsequences = total - seq_length + 1
                for i in range(num_subsequences):
                    if exit == 1000:
                        break
                    exit = exit + 1
                    #print("Subsequence " + str(i+1) + " out of " + str(num_subsequences))
                        
                    #model.zero_grad()

                    subseq_xyz = X_train[:,i:i+seq_length,0:2].float()

                    #subseq_xyz = scaler.fit_transform(subseq_xyz.reshape(-1,1)).reshape(subseq_xyz.shape)      #                      #scaling
                        
                    #subseq_xyz = torch.tensor(subseq_xyz, dtype=torch.float32)             
                    subseq_xyz = subseq_xyz.to(device)
                    #targets_xyz = Y_train[:,i:i+seq_length,0:3].float()

                    targets_xyz = Y_train[:,i:i+seq_length,0:2].float()

                    #targets_xyz = scaler.fit_transform(targets_xyz.reshape(-1,1)).reshape(targets_xyz.shape) #                     #scaling
                        
                    #targets_xyz = torch.tensor(targets_xyz, dtype=torch.float32)
                    targets_xyz = targets_xyz.to(device)

                   # print(subseq_xyz.shape)
                    #print("-----------------------------------------")
                    #print(targets_xyz)
                    """
                    subseq_cont = X_train[:,i:i+seq_length,3:]
                    subseq_cont = subseq_cont.to(device)
                    targets_cont = Y_train[:,i:i+seq_length,3:]
                    targets_cont = targets_cont.to(device)
                    #print(subseq_cont)
                    """

                    model.train()

                    preds_xyz = model(subseq_xyz) # subseq_cont)
                    print(preds_xyz)
                    preds_xyz = preds_xyz.to(device)
                    #pred_conts = preds_cont.to_device()

                    loss = loss_function(preds_xyz,targets_xyz) #+ loss_function(preds_cont, targets_cont)
                    loss = torch.sqrt(loss + eps)
                    l2_lambda = 0.001
                    l2_norm = sum(p.abs().sum() for p in model.parameters())
                    #loss = loss + l2_lambda * l2_norm
                    print("Loss at step " + str(i + 1) + " = " + str(loss))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        print("Loss: " + str(loss))
        if epoch%10 == 0:
            torch.save(model.state_dict(), "models/model_traj/3lstm_3h128_2d_tester_" + "checkpoint_" + str(epoch) + ".pt") 
    #--------------------------------------------------------------------------------------------------------------------------------

    # Testing loop

    # Load model if needed
    #model.load_model('models/simple_model_0.pth')

    print("Begin testing phase.")
    for file in test_files:

        print("Data file: " + file)
        data = dataset_parser(file, seq_length, True)

        dataset = CustomDataset(data, seq_length)

        test_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        #X_test = data[:,:(data.shape[1]-seq_length),0:3]
        #Y_test = data[:,(seq_length):,0:3]
        #X_test = torch.tensor(X_test, dtype=torch.float32)
        #Y_test = torch.tensor(Y_test, dtype=torch.float32)
        #X_test.to(device)
        #Y_test.to(device)

        print("Data ready, start testing.")

        
        for X_test, Y_test in test_dataloader:

            #X_test[:,:,0:2] = torch.tensor(scaler.transform(X_test[:,:,0:2].reshape(-1,2)).reshape(X_test[:,:,0:2].shape))
            #Y_test[:,:,0:2] = torch.tensor(scaler.transform(Y_test[:,:,0:2].reshape(-1,2)).reshape(Y_test[:,:,0:2].shape))

            num_samples, total, input_dim = X_test.size()
            num_subsequences = total - seq_length + 1
            for j in range(num_subsequences):
                #print("Subsequence " + str(j+1) + " out of " + str(num_subsequences))

                model.eval()

                subseq_xyz = X_test[:,j:j+seq_length,0:2].float()
                #subseq_xyz = scaler.transform(subseq_xyz.reshape(-1,1)).reshape(subseq_xyz.shape)

                #subseq_xyz = torch.tensor(subseq_xyz, dtype=torch.float32)     
                subseq_xyz = subseq_xyz.to(device)
                #targets_xyz = Y_test[:,j:j+seq_length,0:3].float()
                targets_xyz = Y_test[:,j:j+seq_length,0:2].float()

                #targets_xyz = scaler.transform(targets_xyz.reshape(-1,1)).reshape(targets_xyz.shape)
                #targets_xyz = torch.tensor(targets_xyz, dtype=torch.float32) 
                targets_xyz = targets_xyz.to(device)
                with torch.no_grad():
                    preds_xyz = model(subseq_xyz)
                    preds_xyz = preds_xyz.to(device)
                    loss = loss_function(preds_xyz,targets_xyz)
                    loss = torch.sqrt(loss + eps)

                print(loss)

    #---------------------------------------------------------------------------------------------------------------------------------

    # Save model if needed

    torch.save(model.state_dict(), "models/model_traj/3lstm_3h128_2d_tester.pt") 
    #dump(scaler, open('scaler.pkl', 'wb'))
    print("Model and scaler saved")