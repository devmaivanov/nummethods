"""A module containing minimization methods."""


def golden(f: callable, a: float, b: float, tolerance: float) -> list:
    """The golden ratio method for one-dimensional minimization."""
    
    GOLDEN_RATIO = ((5**0.5) - 1) / 2
    calculated_values = 0

    while abs(a - b) > tolerance:
        d = GOLDEN_RATIO * (b - a)
        if f(a + d) > f(b - d):
            b = a + d
        else:
            a = b - d
        calculated_values += 2

    return [(a + b) / 2, calculated_values]
