"""A module containing functions for beautiful data/graphs output."""

from matplotlib import pyplot as plt
import numpy as np


def output_method_data(min_value: float, min_point: np.ndarray, iterations: int, calculated_values: int) -> None:
    """A beautiful output of data about the operation of the method."""

    print(f"Minimum value: {min_value}")
    print(f"Minimum point: {min_point[0][0]}; {min_point[1][0]}")
    print(f"Iterations: {iterations - 1}")
    print(f"Calculated values: {calculated_values}")


def plot_relaxation_sequence(X: np.ndarray, Y: np.ndarray , Z: np.ndarray, relaxation_sequence: list) -> None:
    """Plot a beautiful graph of the relaxation sequence."""

    relaxation_sequence_x = []
    relaxation_sequence_y = []
    for point in relaxation_sequence:
        relaxation_sequence_x.append(point[0][0])
        relaxation_sequence_y.append(point[1][0])

    level_lines_min = np.arange(0, 2, 0.5)
    level_lines_med = np.arange(2, 102, 20)
    level_lines_max = np.arange(103, 1503, 100)
    level_lines = np.concatenate([level_lines_min, level_lines_med, level_lines_max])

    plt.grid()
    plt.rcParams["contour.negative_linestyle"] = "solid"
    plt.contour(X, Y, Z, levels=level_lines, colors="blue", linewidths=1)
    plt.plot(relaxation_sequence_x, relaxation_sequence_y, color="black", linewidth=3)
    plt.plot(relaxation_sequence_x[0], relaxation_sequence_y[0], "ro")
    plt.plot(relaxation_sequence_x[len(relaxation_sequence_x)-1], relaxation_sequence_y[len(relaxation_sequence_y)-1], "ro")
    plt.show()
