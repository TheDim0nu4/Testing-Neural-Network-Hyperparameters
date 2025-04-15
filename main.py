import pandas as pd
from data_utils import load_data
from training import train_model
from evaluation import evaluate_model, plot_confusion_matrix
from vizualization import plot_pairplots, print_statistics
from config import LEARNING_RATE
from torch import nn
import torch



data_path = "data/diabetes-dataset.csv"

df = pd.read_csv(data_path)
df['Outcome'] = df['Outcome'].astype(str)

plot_pairplots(df)
print_statistics(df)

train_dataset, X_test_tensor, y_test_tensor, input_size = load_data(df)
output_size = 2



topologies = [ [8], [16], [8, 8], [16, 16],  [8, 8, 8], [16, 16, 16] ]
best_model, best_topology, best_acc = None, None, 0

for topology in topologies:

    model = train_model(input_size, topology, output_size, train_dataset)
    acc, _ = evaluate_model(model, X_test_tensor, y_test_tensor)
    print(f"Topology {topology}. Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_model, best_topology, best_acc = model, topology, acc

print(f"\nBest topology: {best_topology}\n")



optimizers = ["Adam", "SGD", "RMSprop"]
best_optim, best_acc = None, 0

for opt in optimizers:

    model = train_model(input_size, best_topology, output_size, train_dataset, optimizer_name=opt)
    acc, _ = evaluate_model(model, X_test_tensor, y_test_tensor)
    print(f"Optimizer {opt}. Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_model, best_optim, best_acc = model, opt, acc

print(f"\nBest optimizer: {best_optim}\n")



learn_rates = [0.01, 0.001, 0.003, 0.005, 0.05]
best_lr = LEARNING_RATE
best_acc = 0

for lr in learn_rates:

    model = train_model(input_size, best_topology, output_size, train_dataset, optimizer_name=best_optim, lr=lr)
    acc, _ = evaluate_model(model, X_test_tensor, y_test_tensor)
    print(f"Learning rate {lr}. Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_model, best_lr, best_acc = model, lr, acc

print(f"\nBest learning rate: {best_lr}\n")



act_funcs = [nn.ReLU, nn.SELU, nn.Sigmoid, nn.LeakyReLU, nn.Tanh]
act_func_names = { nn.ReLU: "ReLU", nn.SELU: "SELU", nn.Sigmoid: "Sigmoid", nn.LeakyReLU: "LeakyReLU", nn.Tanh: "Tanh" }
best_func, best_acc = None, 0

for func in act_funcs:

    model = train_model(input_size, best_topology, output_size, train_dataset, optimizer_name=best_optim, lr=best_lr, act_func=func)
    acc, pred = evaluate_model(model, X_test_tensor, y_test_tensor)
    print(f"Activation function {act_func_names[func]}. Accuracy: {acc:.4f}")
    
    if acc > best_acc:
        best_model, best_func, best_acc = model, func, acc

print(f"\nBest activation function: {act_func_names[best_func]}\n")



_, pred = evaluate_model(best_model, X_test_tensor, y_test_tensor)
plot_confusion_matrix(y_test_tensor.numpy(), pred.numpy(), ["No Diabetes", "Diabetes"])

torch.save(best_model, "best_model_full.pth")
