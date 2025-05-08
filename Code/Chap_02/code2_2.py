#!/usr/bin/env python
# coding: utf-8

import torch

# Define a tensor with requires_grad=True to track gradients
x = torch.tensor(2.0, requires_grad=True)
print(x)

# Perform operations on the tensor
y = x**2 + 3*x + 1
print(y)

# Compute gradients
y.backward()

# Access the gradients
gradient = x.grad
print(gradient)

x1 = torch.tensor(5.0, requires_grad=True)
print(x1)

# Perform operations on the tensor
y1 = 4*x1**3 + 2*x1**2 + 3*x1 + 1
print(y1)

# Compute gradients
y1.backward()

# Access the gradients
gradient1 = x1.grad
print(gradient1)