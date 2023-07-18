import dataset_parser
import torch
import torch.nn
from lstm_model import LSTModel


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Create model

    window_size = 35

    model = LSTModel(torch.Size([window_size, 3, 1]),window_size,2,False)

    # Prepare parameters

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

	# Prepare the training data

    train_data = dataset_parser('bytes-cafe-2019-02-07_0.json', window_size)

    # Train

    model.run(window_size, train_data, criterion, optimizer) 