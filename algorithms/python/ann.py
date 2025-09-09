# Python program for DVA493 Assignment 1
# Artificial Neural Network

# Libraries
import numpy as np 

print("Artificial Neural Network program starts...")

# Import data:
dataset_path = "../../datasets/maintenance.txt"
data = open(dataset_path)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   dataset info:                                                 #
#   18 columns(0-17) (0-15: features, 16-17: decay coeficients)   #
#   11934(0-11933) examples(samples)                              #
#                                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# data.read()[d][a:b:c] - reads from character a to b in steps of c at row d (0-indexerat)

print(data.readlines()[11933][3:16:1]) 

