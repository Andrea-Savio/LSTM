import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable 
import numpy as np
#from matplotlib import pyplot as plt

from dataset_parser import dataset_parser
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

        self.lstm_xyz = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        #self.lstm_context = nn.LSTM(input_size=2, hidden_size = hidden_size, num_layers = num_layers)
    
        self.fc1 = nn.Linear(3,256)
        self.fc2 = nn.Linear(256,3)

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
        #out_context, (hn_context, cn_context)  = self.lstm_context(context, (h_0cont, c_0cont))
        #print(out_xyz)
        #print(out_context)
        #combined = torch.cat((out_xyz, out_context),dim=2)
        #print(combined)
        out = self.fc1(out_xyz)
        output_xyz = self.fc2(out)

        #output_xyz = self.lstm_xyz1(out[:,0:3])
        #output_cont = self.lstm_context1(out[:,3:])
        #out = self.relu(hn)
        
        #hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        #out = self.relu(hn)
        #out = self.fc_1(out) #first Dense
        #out = self.relu(out) #relu
        #out = self.fc(out) #Final Output
        
        return output_xyz #, output_cont

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
    num_layers = 10
    seq_length = 35
    hidden_size = 3

    model = LSTM_Trainer(input_dim, hidden_size, num_layers, seq_length)
    model = model.to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    
    #---------------------------------------------------------------------------------------------------------------------------------

    # Training loop
    
    print("Begin training phase.")
    for file in train_files:

        with torch.no_grad():
            print("Data file: " + file)
            data = dataset_parser(file, seq_length, True)

            X_train = data[:,:(data.shape[1]-seq_length),:]
            #print(X_train)
            Y_train = data[:,(seq_length):,:]
    
            X_train = torch.tensor(X_train, dtype=torch.float32)
            #X_train = X_train.unsqueeze(dim=0)
            Y_train = torch.tensor(Y_train, dtype=torch.float32)
            #Y_train = Y_train.unsqueeze(dim=0)
            
            #X_train = X_train.to(device)
            #Y_train = Y_train.to(device)
            # Check corresponding sizes
            #print(X_train.size())
            #print(Y_train.size())

            print("Data ready, start training.")
            
            num_samples, total, input_dim = X_train.size()
            num_subsequences = total - seq_length + 1
        
        for i in range(num_subsequences):
            print("Subsequence " + str(i+1) + " out of " + str(num_subsequences))
            
            model.zero_grad()

            subseq_xyz = X_train[:,i:i+seq_length,0:3]
            subseq_xyz = subseq_xyz.to(device)
            targets_xyz = Y_train[:,i:i+seq_length,0:3]
            targets_xyz = targets_xyz.to(device)

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
        data = dataset_parser(file, seq_length, False)

        X_test = data[:,:(data.shape[1]-seq_length),0:3]
        Y_test = data[:,(seq_length):,0:3]
        X_test = torch.tensor(X_test, dtype=torch.float32)
        Y_test = torch.tensor(Y_test, dtype=torch.float32)
        #X_test.to(device)
        #Y_test.to(device)

        print("Data ready, start testing.")

        num_samples, total, input_dim = X_test.size()
        num_subsequences = total - seq_length + 1

        for j in range(num_subsequences):
            print("Subsequence " + str(j+1) + " out of " + str(num_subsequences))

            model.eval()

            subseq_xyz = X_test[:,j:j+seq_length,:]
            subseq_xyz = subseq_xyz.to(device)
            targets_xyz = Y_test[:,j:j+seq_length,:]
            targets_xyz = targets_xyz.to(device)
            with torch.no_grad():
                preds_xyz = model(subseq_xyz)
                preds_xyz = preds_xyz.to(device)
                loss = loss_function(preds_xyz,targets_xyz)

            print(loss)

    #---------------------------------------------------------------------------------------------------------------------------------

    # Save model if needed

    model.save_model("models/model_traj_0.pth")
    print("Model saved") 