# Python program for DVA493 Assignment 1
# Artificial Neural Network using NumPy matrices and Matplotlib

# Libraries
import numpy as np
import matplotlib.pyplot as plt

print("Artificial Neural Network program starts...")

# Import data:
dataset_path = "../../datasets/maintenance.txt"
# datasets:
trainingset = []
validationset = []
testset = []
with open(dataset_path, "r") as datafile:
    rowIndex = 0
    for example in datafile: # each example is its own line
        example = example.strip() #removes "\n"
        row = example.split("  ") # splits the datapoints with its spacing
        dataValue = [float(v) for v in row if v] # conversion

        if rowIndex < 5968:
            trainingset.append(dataValue) # => trainingset[column][row]
        elif rowIndex < 8951:
            validationset.append(dataValue) # => validationset[column][row]
        else:
            testset.append(dataValue) # => testset[column][row]
        rowIndex = rowIndex + 1
# Transposes the matrices to achieve nicer data structures using numpy arrays: 
# dataset[column][row] to dataset[row][column]
trainingset = np.array(trainingset)
validationset = np.array(validationset)
testset = np.array(testset)
trainingset = trainingset.T
validationset = validationset.T
testset = testset.T
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#      dataset info:                                                    #
#   18 columns(0-17) (0-15: features, 16-17: decay coeficients)         #
#   11934(0-11933) examples(samples)                                    #
#   0-5961:       trainingset[18][5968]           (slightly over 50%)   #
#   5962-8942:    validationset[18][2983]                      (~25%)   #
#   8943-11933:   testset[18][2983]                            (~25%)   #
#                                                                       #
#      Features as GT measures(in order):                               #
#   0: Lever position (lp) [ ]                                          #
#   1: Ship speed (v) [knots]                                           #
#   2: Gas Turbine (GT) shaft torque (GTT) [kN m]                       #
#   3: GT rate of revolutions (GTn) [rpm]                               #
#   4: Gas Generator rate of revolutions (GGn) [rpm]                    #
#   5: Starboard Propeller Torque (Ts) [kN]                             #
#   6: Port Propeller Torque (Tp) [kN]                                  #
#   7: Hight Pressure (HP) Turbine exit temperature (T48) [C]           #
#   8: GT Compressor inlet air temperature (T1) [C]                     #
#   9: GT Compressor outlet air temperature (T2) [C]                    #
#   10: HP Turbine exit pressure (P48) [bar]                            #
#   11: GT Compressor inlet air pressure (P1) [bar]                     #
#   12: GT Compressor outlet air pressure (P2) [bar]                    #
#   13: GT exhaust gas pressure (Pexh) [bar]                            #
#   14: Turbine Injecton Control (TIC) [%]                              #
#   15: Fuel flow (mf) [kg/s]                                           #
#                                                                       #
#     Decay coefficients(in order):                                     #
#  16: GT Compressor decay state coefficient                            #
#  17: GT Turbine decay state coefficient                               #
#                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Artificial Neural Network Implementation:

