# Python program in preparation for Assignment 1 - DVA493
# Artificial Neuron or "preceptron" using NumPy matrices and Matplotlib

# Libraries
import numpy as np
import matplotlib.pyplot as plt

print("Preceptron program starts...")

# Data (Taken from a youtube example):
PSA = np.array([ 3.8, 3.4, 2.9, 2.8, 2.7, 2.1, 1.6, 2.5, 2.0, 1.7, 1.4, 1.2, 0.9, 0.8 ]).reshape(-1, 1)  # 14 Inputs
Status = np.array([ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 ])                                          # Labels
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   dataset info:           https://www.youtube.com/watch?v=XxZ0BibMTjw   #
#         PSA - ug/L                                                      #
#         Status - Cancer(1) or Healthy(0)                                #
#   Data are reshaped to achive a 2D matrix with 14 rows and 1 column     #
#   Status are set as a scalar                                            #
#                                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Preceptron Implementation:
# Weight and Bias:
W = np.random.randn(1, 1)
b = np.random.randn(1)
