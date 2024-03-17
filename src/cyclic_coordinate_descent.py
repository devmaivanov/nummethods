# Software implementation of the cyclic coordinate descent method

import minimization
import outputm

import numdifftools as nd
import numpy as np
import sys


# Target function
def f(var: list) -> float:
    return ((1.-var[0])**2) + 100.*((var[1]-(var[0]**2))**2)


# Initial data
tolerance = float(sys.argv[1])
starting_point = np.array([[float(sys.argv[2])], [float(sys.argv[3])]])

# Output data
relaxation_sequence = []
function_values = []
calculated_values: int = 0
iterations: int = 0

# Adding a starting point to relaxation sequence
relaxation_sequence.append(starting_point)
iterations += 1

# Calculating the value of the target function at the starting point
function_values.append(f([starting_point[0][0], starting_point[1][0]]))
calculated_values += 1

# Creating a basis of n vectors
e0 = np.array([[1.], [0.]])
e1 = np.array([[0.], [1.]])
basis = [e0, e1]

# Setting the values in order to get into the loop
current_f_value = 1
new_f_value = 0
alpha = 1

# Stop criteria
while abs(new_f_value - current_f_value) > tolerance or abs(alpha) > tolerance:
    current_point = relaxation_sequence[iterations-1]
    current_f_value = function_values[iterations-1]

    # Determining direction of which basic vector we are moving
    if iterations % len(basis) == 0:
        e = basis[0]
    else:
        e = basis[1]

    # The formula for one-dimensional minimization, from which we find phi
    def phi(alpha):
        x = current_point[0][0] + e[0][0]*alpha
        y = current_point[1][0] + e[1][0]*alpha
        return f([x, y])
    
    # One-dimensional minimization
    golden_tolerance = 0.0001
    golden_data = minimization.golden(f=phi, a=-50., b=50., tolerance=golden_tolerance)
    alpha = golden_data[0]
    calculated_values += golden_data[1]

    # Finding a new point of relaxation sequence
    new_point = np.array([[0.], [0.]])
    new_point[0][0] = current_point[0][0] + e[0][0]*alpha
    new_point[1][0] = current_point[1][0] + e[1][0]*alpha
    relaxation_sequence.append(new_point)
    iterations += 1

    # Getting the value of the target function at a new point
    new_f_value = f([new_point[0][0], new_point[1][0]])
    function_values.append(new_f_value)
    calculated_values += 1


# Output of information about the operation of the method
outputm.output_method_data(function_values[iterations-1], relaxation_sequence[iterations-1], iterations, calculated_values)

# Graph axis settings
x_axis = np.arange(-5, 5, 0.01)
y_axis = np.arange(-2.5, 12.5, 0.01)

# Creating a coordinate grid of the graph
X, Y = np.meshgrid(x_axis, y_axis)
Z = f([X, Y])

# Plotting a graph of the relaxation sequence 
outputm.plot_relaxation_sequence(X, Y, Z, relaxation_sequence)
