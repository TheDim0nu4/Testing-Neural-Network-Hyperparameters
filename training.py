import torch
from torch import optim
from torch.utils.data import DataLoader
from mlp import MLP
from config import set_seed, EPOCHS, BATCH_SIZE



def train_model(input_size, topology, output_size, train_dataset, optimizer_name="Adam", lr=0.003, act_func=None):

    set_seed()

    if act_func is None:
        from torch import nn
        act_func = nn.ReLU

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = MLP(input_size, topology, output_size, activation_func=act_func)

    criterion = torch.nn.CrossEntropyLoss()
    

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer")


    for epoch in range(EPOCHS):

        model.train()

        for inputs, labels in train_loader:
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    return model
