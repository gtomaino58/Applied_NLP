import torch
import torch.nn as nn
import torchsummary

#Define a simple FNN model
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # First layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second layer
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Pass input through first layer
        x = self.fc2(x)              # Pass through second layer
        return x                     # Return output

#Example usage

if __name__ == "__main__":

    # Define input size, hidden layer size, and output size
    input_size = 64  # Number of input features
    hidden_size = 128  # Number of neurons in the hidden layer
    output_size = 10  # Number of output features

    # Create an instance of the model
    model = SimpleNet(input_size, hidden_size, output_size)

    # Print the model summary using torchsummary
    summary = torchsummary.summary(model, (input_size,))  # Import summary function from torchsummary
                                       # Print the model summary with input size
