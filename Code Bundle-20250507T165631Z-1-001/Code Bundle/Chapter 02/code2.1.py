#!/usr/bin/env python
# coding: utf-8

import torch
# Create a tensor from a list
tensor_a = torch.tensor([1, 2, 3])
# Create a tensor filled with zeros
tensor_zeros = torch.zeros(2, 3)
print(tensor_zeros)
# Create a tensor filled with ones
tensor_ones = torch.ones(2, 3)
# Create a tensor with random values
tensor_random = torch.rand(2, 3)
print(tensor_random)
# Element-wise addition
result = tensor_a + tensor_a
print(result)
# Element-wise multiplication
result = tensor_a * 2
print(result)
# Matrix multiplication
matrix_a = torch.tensor([[1, 2], [3, 4]])
print(matrix_a)
matrix_b = torch.tensor([[5, 6], [7, 8]])
print(matrix_b)
result = torch.matmul(matrix_a, matrix_b)
print(result)