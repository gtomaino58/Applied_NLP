import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torchsummary
import torchvision

import numpy as np
import matplotlib.pyplot as plt

import os
import random
import time

# Vamos a usar MNIST para nmostrar como funciona una FNN simple para clasificar dígitos
# MNIST es un dataset de dígitos escritos a mano

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Vamos a ver que tenemos en mnist y mnist_test
print(mnist_train)
print()
print(mnist_test)
print()

print(mnist_train.data.shape)  # (60000, 28, 28)
print()
print(mnist_test.data.shape)  # (10000, 28, 28)
print()

print(mnist_train.data[0].shape)  # (28, 28)
print(mnist_train.data[0])  # 0-9
print()
print(mnist_test.data[0].shape)  # (28, 28)
print(mnist_test.data[0])  # 0-9
print()

# Vamos a definir una FNN para clasificar los dígitos
# La red tendrá 3 capas ocultas y una capa de salida, la función de activación será ReLU
# La función de pérdida será CrossEntropyLoss, el optimizador será Adam, la tasa de aprendizaje será 0.001
# El tamaño del batch será 32, el número de épocas será 10
# El número de clases será 10 (0-9), el número de entradas será 28*28=784, el número de salidas será 10 (0-9)
# El número de neuronas en la primera capa oculta será 128, el número de neuronas en la segunda capa oculta será 64
# El número de neuronas en la tercera capa oculta será 32, el número de neuronas en la capa de salida será 10 (0-9)
# La función de activación de la capa de salida será Softmax
    
# Vamos a definir la red

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x
    
# Definimos los hiperparámetros
input_size = 28 * 28  # 784
hidden_size1 = 128
hidden_size2 = 64
hidden_size3 = 32
output_size = 10  # 0-9

num_epochs = 10
batch_size = 64
learning_rate = 0.001

num_classes = 10  # 0-9

# Definimos el dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Definimos la red
model = FNN(input_size, hidden_size1, hidden_size2, hidden_size3, output_size).to(device)
print(model)
print()

# Definimos la función de pérdida y el optimizador
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Definimos el DataLoader
train_loader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


# Definimos la función de entrenamiento
def train(model, train_loader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.view(-1, input_size).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(train_loader), accuracy


# Definimos la función de prueba
def test(model, test_loader, loss_function, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, input_size).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(test_loader), accuracy


# Entrenamos la red
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
start_time = time.time()
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, loss_function, optimizer, device)
    test_loss, test_accuracy = test(model, test_loader, loss_function, device)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

end_time = time.time()
print(f'Training time: {end_time - start_time:.2f} seconds')


# Graficamos la pérdida y la precisión
def plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    # Precisión
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epochs')
    plt.legend()

    plt.show()

plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies)

