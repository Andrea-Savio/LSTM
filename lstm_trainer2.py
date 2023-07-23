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

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Define how data flows into the network
    
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size, device=x.device)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size, device=x.device)) #internal state
        # Propagate input through LSTM
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        
        #out = self.relu(hn)
        
        #hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        #out = self.relu(hn)
        #out = self.fc_1(out) #first Dense
        #out = self.relu(out) #relu
        #out = self.fc(out) #Final Output
        
        return output

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
    num_layers = 3
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

            #data[data==0] = np.nan

            X_train = data[:,:(data.shape[1]-seq_length),:]
            Y_train = data[:,(seq_length):,:]
            X_train = torch.tensor(X_train, dtype=torch.float32)
            Y_train = torch.tensor(Y_train, dtype=torch.float32)
            
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

            subseq = X_train[:,i:i+seq_length,:]
            subseq = subseq.to(device)
            targets = Y_train[:,i:i+seq_length,:]
            targets = targets.to(device)

            model.train()

            preds = model(subseq)
            preds = preds.to(device)

            loss = loss_function(preds,targets)
            print("Loss at step " + str(i + 1) + " = " + str(loss))
            loss.backward()
            optimizer.step()
    
    #--------------------------------------------------------------------------------------------------------------------------------

    # Testing loop

    #model.load_model('models/simple_model_0.pth')

    print("Begin testing phase.")
    for file in test_files:

        print("Data file: " + file)
        data = dataset_parser(file, seq_length, False)

        X_test = data[:,:(data.shape[1]-seq_length),:]
        Y_test = data[:,(seq_length):,:]
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

            subseq = X_test[:,j:j+seq_length,:]
            subseq = subseq.to(device)
            targets = Y_test[:,j:j+seq_length,:]
            targets = targets.to(device)
            with torch.no_grad():
                preds = model(subseq)
                preds = preds.to(device)
                loss = loss_function(preds,targets)

            print(loss)

    #---------------------------------------------------------------------------------------------------------------------------------

    # Save model if needed

    model.save_model("models/simple_model_0.pth") 