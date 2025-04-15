import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



def evaluate_model(model, X_test_tensor, y_test_tensor):

    model.eval()

    with torch.no_grad():

        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_test_tensor).sum().item()
        accuracy = correct / y_test_tensor.size(0)


    return accuracy, predicted



def plot_confusion_matrix(y_true, y_pred, class_names):
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
