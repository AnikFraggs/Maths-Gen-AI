import numpy as np
from scipy import integrate

DATASET = [
    ("x^2", lambda x: x**2, 0, 1, 1/3, ["polynomial", "smooth"]),
    ("x^4 - 3x^2 + 1", lambda x: x**4 - 3*x**2 + 1, -1, 2, 6.6, ["polynomial", "smooth"]),
    ("sin(x)", np.sin, 0, np.pi, 2.0, ["trigonometric", "smooth", "oscillatory"]),
    ("sin(50x)", lambda x: np.sin(50*x), 0, np.pi, 0.0, ["oscillatory", "high_frequency"]),
    ("e^(-x^2)", lambda x: np.exp(-x**2), -3, 3, np.sqrt(np.pi), ["exponential", "smooth", "gaussian_shape"]),
    ("1/sqrt(x)", lambda x: 1/np.sqrt(np.maximum(x,1e-15)), 0, 1, 2.0, ["algebraic", "endpoint_singularity"]),
    ("ln(x)", np.log, 1, np.e, 1.0, ["logarithm", "smooth"]),
    ("1/(1+x^2)", lambda x: 1/(1+x**2), 0, 1, np.pi/4, ["rational", "smooth"]),
    ("exp(-100*(x-0.5)^2)", lambda x: np.exp(-100*(x-0.5)**2), 0, 1, 0.1772, ["peaked", "gaussian_shape"]),
    ("Heaviside at 0.5", lambda x: 1.0 if x > 0.5 else 0.0, 0, 1, 0.5, ["piecewise", "discontinuous"]),
    ("abs(sin(x))", lambda x: np.abs(np.sin(x)), 0, 2*np.pi, 4.0, ["piecewise", "non_smooth"]),
    ("cos(x^2)", lambda x: np.cos(x**2), 0, 5, 0.5, ["oscillatory", "chirp"]),
]
