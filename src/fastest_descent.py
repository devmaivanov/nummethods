# Software implementation of the fastest descent method

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

# Gradient of target function
df = nd.Gradient(f)

# Setting the gradient value in order to get into the loop
gradient = np.array([[1.], [1.]])

# Stop criteria
while ((gradient[0][0]**2) + (gradient[1][0]**2))**0.5 > tolerance:
    current_point = relaxation_sequence[iterations-1]

    # Calculating the gradient at the current point
    gradient[0][0], gradient[1][0] = df([current_point[0][0], current_point[1][0]])
    calculated_values += 2

    # The formula for one-dimensional minimization, from which we find xi
    def psi(xi):
        x = current_point[0][0] + (-1)*gradient[0][0]*abs(xi)
        y = current_point[1][0] + (-1)*gradient[1][0]*abs(xi)
        return f([x, y])
    
    # One-dimensional minimization
    golden_tolerance = 0.0001
    golden_data = minimization.golden(f=psi, a=0., b=0.1, tolerance=golden_tolerance)
    xi = abs(golden_data[0])
    calculated_values += golden_data[1]

    # Finding a new point of relaxation sequence
    new_point = np.array([[0.], [0.]])
    new_point[0][0] = current_point[0][0] + (-1)*gradient[0][0]*xi
    new_point[1][0] = current_point[1][0] + (-1)*gradient[1][0]*xi
    relaxation_sequence.append(new_point)
    iterations += 1

    # Getting the value of the target function at a new point
    function_values.append(f([new_point[0][0], new_point[1][0]]))

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
