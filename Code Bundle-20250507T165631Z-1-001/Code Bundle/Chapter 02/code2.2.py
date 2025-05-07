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