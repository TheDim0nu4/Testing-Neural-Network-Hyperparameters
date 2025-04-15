import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from config import set_seed



def load_data(df):

    set_seed()
    
    X = df.drop(columns=['Outcome'])
    Y = df[['Outcome']].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).squeeze()
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).squeeze()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)


    return train_dataset, X_test_tensor, y_test_tensor, X.shape[1]
