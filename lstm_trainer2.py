import torch
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable 
import numpy as np
from pickle import dump

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score                         
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
#from matplotlib import pyplot as plt

from dataset_parser import dataset_parser
from dataset_parser import CustomDataset
from dataset_parser import create_subsequence

class LSTM_Trainer(nn.Module):

    # Model definition

    def __init__(self, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_Trainer, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.output_size = input_size

        self.lstm_xyz = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5)
        #self.lstm_context = nn.LSTM(input_size=2, hidden_size = hidden_size, num_layers = num_layers)
    
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32,3)
        self.dropout = nn.Dropout(p=0.5)
        #self.linear = nn.Linear(self.hidden_size, self.output_size)

        #self.lstm_xyz1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        #self.lstm_context1 = nn.LSTM(input_size=2, hidden_size = hidden_size, num_layers = num_layers)

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Define how data flows into the network
    
    def forward(self, xyz): #, context)
        h_0xyz = Variable(torch.zeros(self.num_layers, xyz.size(1), self.hidden_size, device=xyz.device)) #hidden state
        c_0xyz = Variable(torch.zeros(self.num_layers, xyz.size(1), self.hidden_size, device=xyz.device)) #internal state

        #h_0cont = Variable(torch.zeros(self.num_layers, context.size(1), self.hidden_size, device=context.device)) 
        #c_0cont = Variable(torch.zeros(self.num_layers, context.size(1), self.hidden_size, device=context.device))
        
        # Propagate input through LSTM
        
        out_xyz, (hn_xyz, cn_xyz) = self.lstm_xyz(xyz, (h_0xyz, c_0xyz)) #lstm with input, hidden, and internal state
        #out_xyz, _ = rnn_utils.pad_packed_sequence(packed_out_xyz, batch_first=True)

        #out_context, (hn_context, cn_context)  = self.lstm_context(context, (h_0cont, c_0cont))
        #print(out_xyz)
        #print(out_context)
        #combined = torch.cat((out_xyz, out_context),dim=2)
        #print(combined)
        
        out = self.fc1(out_xyz)
        out = self.fc2(out)
        output_xyz = self.fc3(out)
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

    train_files = ['packard-poster-session-2019-03-20_1.json', 'serra-street-2019-01-30_0.json', 'bytes-cafe-2019-02-07_0.json', 'nvidia-aud-2019-01-25_0.json', 'discovery-walk-2019-02-28_0.json', 'memorial-court-2019-03-16_0.json', 'gates-foyer-2019-01-17_0.json', 'tressider-2019-03-16_2.json', 'huang-basement-2019-01-25_0.json', 'gates-basement-elevators-2019-01-17_1.json', 'discovery-walk-2019-02-28_1.json', 'tressider-2019-04-26_2.json', 'svl-meeting-gates-2-2019-04-08_0.json', 'meyer-green-2019-03-16_1.json', 'gates-to-clark-2019-02-28_1.json', 'svl-meeting-gates-2-2019-04-08_1.json', 'gates-ai-lab-2019-02-08_0.json', 'tressider-2019-03-16_0.json', 'stlc-111-2019-04-19_0.json', 'tressider-2019-04-26_0.json', 'gates-to-clark-2019-02-28_0.json', 'gates-ai-lab-2019-04-17_0.json', 'huang-2-2019-01-25_0.json', 'tressider-2019-04-26_1.json', 'stlc-111-2019-04-19_1.json', 'lomita-serra-intersection-2019-01-30_0.json', 'hewlett-class-2019-01-23_1.json', 'cubberly-auditorium-2019-04-22_1.json', 'hewlett-packard-intersection-2019-01-24_0.json', 'tressider-2019-03-16_1.json', 'clark-center-2019-02-28_1.json', 'huang-lane-2019-02-12_0.json', 'tressider-2019-04-26_3.json', 'nvidia-aud-2019-04-18_1.json', 'huang-intersection-2019-01-22_0.json', 'packard-poster-session-2019-03-20_2.json', 'food-trucks-2019-02-12_0.json', 'packard-poster-session-2019-03-20_0.json', 'outdoor-coupa-cafe-2019-02-06_0.json', 'forbes-cafe-2019-01-22_0.json', 'nvidia-aud-2019-04-18_0.json', 'meyer-green-2019-03-16_0.json', 'quarry-road-2019-02-28_0.json', 'cubberly-auditorium-2019-04-22_0.json', 'nvidia-aud-2019-04-18_2.json', 'hewlett-class-2019-01-23_0.json', 'jordan-hall-2019-04-22_0.json', 'indoor-coupa-cafe-2019-02-06_0.json', 'clark-center-intersection-2019-02-28_0.json', 'huang-2-2019-01-25_1.json', 'stlc-111-2019-04-19_2.json', 'gates-159-group-meeting-2019-04-03_0.json', 'gates-basement-elevators-2019-01-17_0.json', 'clark-center-2019-02-28_0.json']
    test_files = ['serra-street-2019-01-30_0.json', 'nvidia-aud-2019-01-25_0.json', 'discovery-walk-2019-02-28_0.json', 'gates-foyer-2019-01-17_0.json', 'tressider-2019-03-16_2.json', 'discovery-walk-2019-02-28_1.json', 'meyer-green-2019-03-16_1.json', 'tressider-2019-04-26_0.json', 'gates-to-clark-2019-02-28_0.json', 'gates-ai-lab-2019-04-17_0.json', 'tressider-2019-04-26_1.json', 'stlc-111-2019-04-19_1.json', 'lomita-serra-intersection-2019-01-30_0.json', 'hewlett-class-2019-01-23_1.json', 'cubberly-auditorium-2019-04-22_1.json', 'tressider-2019-04-26_3.json', 'nvidia-aud-2019-04-18_1.json', 'huang-intersection-2019-01-22_0.json', 'food-trucks-2019-02-12_0.json', 'outdoor-coupa-cafe-2019-02-06_0.json', 'quarry-road-2019-02-28_0.json', 'nvidia-aud-2019-04-18_2.json', 'hewlett-class-2019-01-23_0.json', 'indoor-coupa-cafe-2019-02-06_0.json', 'huang-2-2019-01-25_1.json', 'stlc-111-2019-04-19_2.json', 'gates-basement-elevators-2019-01-17_0.json']

    input_dim = 3
    num_layers = 2
    seq_length = 35 #35
    hidden_size = 128
    num_epochs = 1
    scaled = False

    model = LSTM_Trainer(input_dim, hidden_size, num_layers, seq_length)
    model = model.to(device)
    loss_function = nn.L1Loss()
    optimizer = optim.Adagrad(model.parameters(),lr=1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    #---------------------------------------------------------------------------------------------------------------------------------

    # Training loop
    
    print("Begin training phase.")
    for file in train_files:

        with torch.no_grad():
            print("Data file: " + file)
            data, lengths = dataset_parser(file, seq_length, True)

            #packed_sequences = rnn_utils.pack_padded_sequence(data,
                                                  #lengths, batch_first=True, enforce_sorted=False)
            dataset = CustomDataset(data, seq_length)

            train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

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
        for epoch in range(num_epochs):
            print("Epoch: " + str(epoch))
            for X_train, Y_train in train_dataloader:
                #X_train = rnn_utils.pack_padded_sequence(X_train,
                #                                  lengths[0:63], batch_first=True, enforce_sorted=False)
                #Y_train = rnn_utils.pack_padded_sequence(Y_train,
                #                                  lengths[0:63], batch_first=True, enforce_sorted=False)
                
                if not scaled:
                    X_train[:,:,0:3] = torch.tensor(scaler.fit_transform(X_train[:,:,0:3].reshape(-1,1)).reshape(X_train[:,:,0:3].shape), dtype=torch.float32)
                    Y_train[:,:,0:3] = torch.tensor(scaler.transform(Y_train[:,:,0:3].reshape(-1,1)).reshape(Y_train[:,:,0:3].shape), dtype=torch.float32)
                    scaled = True

                X_train[:,:,0:3] = torch.tensor(scaler.transform(X_train[:,:,0:3].reshape(-1,1)).reshape(X_train[:,:,0:3].shape), dtype=torch.float32)
                Y_train[:,:,0:3] = torch.tensor(scaler.transform(Y_train[:,:,0:3].reshape(-1,1)).reshape(Y_train[:,:,0:3].shape), dtype=torch.float32)

                num_samples, total, input_dim = X_train.size()
                num_subsequences = total - seq_length + 1
                for i in range(num_subsequences):
                    print("Subsequence " + str(i+1) + " out of " + str(num_subsequences))
                    
                    model.zero_grad()

                    subseq_xyz = X_train[:,i:i+seq_length,0:3].float()

                    #subseq_xyz = scaler.fit_transform(subseq_xyz.reshape(-1,1)).reshape(subseq_xyz.shape)      #                      #scaling
                    
                    #subseq_xyz = torch.tensor(subseq_xyz, dtype=torch.float32)             
                    subseq_xyz = subseq_xyz.to(device)
                    targets_xyz = Y_train[:,i:i+seq_length,0:3].float()

                    #targets_xyz = scaler.fit_transform(targets_xyz.reshape(-1,1)).reshape(targets_xyz.shape) #                     #scaling
                    
                    #targets_xyz = torch.tensor(targets_xyz, dtype=torch.float32)
                    targets_xyz = targets_xyz.to(device)

                    #print(subseq_xyz.shape)
                    #print("-----------------------------------------")
                    #print(targets_xyz.shape)
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
                    l2_lambda = 0.001
                    l2_norm = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + l2_lambda * l2_norm
                    print("Loss at step " + str(i + 1) + " = " + str(loss))
                    loss.backward()
                    optimizer.step()
    
    #--------------------------------------------------------------------------------------------------------------------------------

    # Testing loop

    # Load model if needed
    #model.load_model('models/simple_model_0.pth')

    print("Begin testing phase.")
    for file in test_files:

        print("Data file: " + file)
        data, lengths = dataset_parser(file, seq_length, False)

        dataset = CustomDataset(data, seq_length)

        test_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        #X_test = data[:,:(data.shape[1]-seq_length),0:3]
        #Y_test = data[:,(seq_length):,0:3]
        #X_test = torch.tensor(X_test, dtype=torch.float32)
        #Y_test = torch.tensor(Y_test, dtype=torch.float32)
        #X_test.to(device)
        #Y_test.to(device)

        print("Data ready, start testing.")

        
        for X_test, Y_test in test_dataloader:

            X_test[:,:,0:3] = torch.tensor(scaler.transform(X_test[:,:,0:3].reshape(-1,1)).reshape(X_test[:,:,0:3].shape))
            Y_test[:,:,0:3] = torch.tensor(scaler.transform(Y_test[:,:,0:3].reshape(-1,1)).reshape(Y_test[:,:,0:3].shape))

            num_samples, total, input_dim = X_test.size()
            num_subsequences = total - seq_length + 1
            for j in range(num_subsequences):
                print("Subsequence " + str(j+1) + " out of " + str(num_subsequences))

                model.eval()

                subseq_xyz = X_test[:,j:j+seq_length,0:3].float()
                #subseq_xyz = scaler.transform(subseq_xyz.reshape(-1,1)).reshape(subseq_xyz.shape)

                subseq_xyz = torch.tensor(subseq_xyz, dtype=torch.float32)     
                subseq_xyz = subseq_xyz.to(device)
                targets_xyz = Y_test[:,j:j+seq_length,0:3].float()

                #targets_xyz = scaler.transform(targets_xyz.reshape(-1,1)).reshape(targets_xyz.shape)
                targets_xyz = torch.tensor(targets_xyz, dtype=torch.float32) 
                targets_xyz = targets_xyz.to(device)
                with torch.no_grad():
                    preds_xyz = model(subseq_xyz)
                    preds_xyz = preds_xyz.to(device)
                    loss = loss_function(preds_xyz,targets_xyz)

                print(loss)

    #---------------------------------------------------------------------------------------------------------------------------------

    # Save model if needed

    model.save_model("models/model_scaled_3linear_128hidden_single_epoch_no_padding_2.pth") 
    dump(scaler, open('scaler.pkl', 'wb'))
    print("Model and scaler saved")