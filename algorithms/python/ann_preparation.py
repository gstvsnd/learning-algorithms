# Python program in preparation for Assignment 1 - DVA493
# Artificial Neuron or "preceptron" using NumPy matrices and Matplotlib

# ----- Dataset & Input -----
# ----- Forward Pass -----
# ----- Activation Functions -----
# ----- Loss Function -----
# ----- Multi Layer Forward Pass -----
# ----- Backpropagation & Weight Updates -----
# ----- Training Loop -----


# Libraries
import numpy as np
import matplotlib.pyplot as plt

print("Preceptron program starts...")

# Data: (Taken from a youtube example)
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

np.random.seed(0)

class Layer_d:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_sigmoid: # ReLU, SoftMax
    def forward(self, z):
        self.output = 1/(1 + np.exp(-z))
        #return 1/(1 + np.exp(-z))

class Activation_ReLU: # Här upplever jag begränsad förståelse!
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax: # Här upplever jag begränsad förståelse!
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses) #batch loss
        return data_loss

class Loss_CategoricalCrossentrophy(Loss): # Här upplever jag begränsad förståelse!
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) 

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

#X, y = spiral_data(samples=100, classes=3)


#X = [[1, 2, 3, 2.5], 
#    [2, 5, -1, 2], 
#    [-1.5, 2.7, 3.3, -0.8]]   # shape: (4, 3)

X = np.array([
    [1, 1],
    [2, 1],
    [1.5, 2],
    [5, 5],
    [6, 5],
    [5.5, 6],
    [9, 1],
    [8, 2],
    [9, 2]
])


# Labels (y): class indices (0, 1, 2)
y = np.array([0, 0, 0,   # first 3 belong to class 0
              1, 1, 1,   # next 3 belong to class 1
              2, 2, 2])  # last 3 belong to class 2

# LayersACA
#layer1 = Layer_d(2, 9)
#layer1.forward(X)
print(f"Input: {X}") # shape = (9, 2)

dense1 = Layer_d(2, 9) # 2 inputs and 9 neurons
activation1 = Activation_ReLU()
#print(f"Layer 1: {activation1.output[:5]}") # shape = (1, 1) - result of the dot-product

dense2 = Layer_d(9, 3)
activation2 = Activation_Softmax()
#print(f"Layer 2: {activation2.output[:5]}") # shape = (1, 1)

# Forward pass:
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print("Probabilities:\n", activation2.output[:5])

loss_funktion = Loss_CategoricalCrossentrophy()
loss = loss_funktion.calculate(activation2.output, y)

print("Loss: ", loss)

# We want a small loss!



