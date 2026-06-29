from dataset import DATASET
from methods import METHODS
DATASET = [
    # ── Polynomials ───────────────────────────────────────────────
    ("x^2",            lambda x: x**2,                      0, 1,   1/3,
     ["polynomial", "smooth", "no_singularity"]),
    ("x^4 - 3x^2 + 1", lambda x: x**4 - 3*x**2 + 1,       -1, 2,  (32/5 - 8 + 2) - (-1/5 + 1 - 1),
     ["polynomial", "smooth", "oscillatory"]),
    ("x^7",            lambda x: x**7,                       0, 1,  1/8,
     ["polynomial", "smooth", "no_singularity"]),
 
    # ── Trigonometric ─────────────────────────────────────────────
    ("sin(x)",         np.sin,                               0, np.pi, 2.0,
     ["trigonometric", "smooth", "oscillatory"]),
    ("cos(x)",         np.cos,                               0, np.pi, 0.0,
     ["trigonometric", "smooth", "oscillatory"]),
    ("sin(5x)",        lambda x: np.sin(5*x),                0, np.pi, 2/5,
     ["trigonometric", "smooth", "high_frequency"]),
    ("sin(x)*cos(x)",  lambda x: np.sin(x)*np.cos(x),       0, np.pi/2, 0.5,
     ["trigonometric", "smooth"]),
 
    # ── Exponentials ─────────────────────────────────────────────
    ("e^x",            np.exp,                               0, 1,  np.e - 1,
     ["exponential", "smooth", "monotone"]),
    ("e^(-x^2)",       lambda x: np.exp(-x**2),              -3, 3,  np.sqrt(np.pi)*( 1 - 2*1.1254e-5),  # ≈ √π
     ["exponential", "smooth", "gaussian_shape"]),
    ("e^(-x)*sin(x)",  lambda x: np.exp(-x)*np.sin(x),      0, np.pi, (1 + np.exp(-np.pi))/2,
     ["exponential", "smooth", "oscillatory", "decay"]),
    ("x*e^x",          lambda x: x*np.exp(x),                0, 1,  1.0,
     ["exponential", "smooth"]),
 
    # ── Logarithms ───────────────────────────────────────────────
    ("ln(x)",          np.log,                               1, np.e, 1.0,
     ["logarithm", "smooth", "monotone"]),
    ("ln(1+x)",        lambda x: np.log(1+x),                0, 1,  2*np.log(2)-1,
     ["logarithm", "smooth"]),
    ("x*ln(x)",        lambda x: x*np.log(x),                1, 2,  2*np.log(2) - 3/4,
     ["logarithm", "smooth"]),
 
    # ── Algebraic / Rational ──────────────────────────────────────
    ("1/(1+x^2)",      lambda x: 1/(1+x**2),                 0, 1,  np.pi/4,
     ["rational", "smooth", "monotone"]),
    ("1/(1+x)",        lambda x: 1/(1+x),                    0, 1,  np.log(2),
     ["rational", "smooth", "monotone"]),
    ("sqrt(x)",        np.sqrt,                              0, 1,  2/3,
     ["algebraic", "smooth", "endpoint_singularity_derivative"]),
    ("sqrt(1-x^2)",    lambda x: np.sqrt(np.clip(1-x**2,0,None)), -1, 1, np.pi/2,
     ["algebraic", "smooth", "endpoint_singularity_derivative"]),
    ("1/sqrt(x)",      lambda x: 1/np.sqrt(np.maximum(x,1e-15)), 0, 1, 2.0,
     ["algebraic", "endpoint_singularity", "singular"]),
 
    # ── Highly oscillatory ────────────────────────────────────────
    ("sin(10x)",       lambda x: np.sin(10*x),               0, np.pi, (1-np.cos(10*np.pi))/10,
     ["trigonometric", "high_frequency", "oscillatory"]),
    ("sin(50x)",       lambda x: np.sin(50*x),               0, np.pi, (1-np.cos(50*np.pi))/50,
     ["trigonometric", "very_high_frequency", "oscillatory"]),
    ("cos(x^2)",       lambda x: np.cos(x**2),               0, 5,
     integrate.quad(lambda x: np.cos(x**2), 0, 5)[0],
     ["oscillatory", "chirp", "high_frequency"]),
 
    # ── Near-singular / peaked ────────────────────────────────────
    ("1/(x^0.5+0.01)", lambda x: 1/(np.sqrt(np.maximum(x,0))+0.01), 0, 1,
     integrate.quad(lambda x: 1/(np.sqrt(x)+0.01), 0, 1)[0],
     ["near_singular", "endpoint_singularity"]),
    ("exp(-100*(x-0.5)^2)", lambda x: np.exp(-100*(x-0.5)**2), 0, 1,
     integrate.quad(lambda x: np.exp(-100*(x-0.5)**2), 0, 1)[0],
     ["peaked", "smooth", "gaussian_shape"]),
    ("1/(x^2+0.001)",  lambda x: 1/(x**2+0.001),             0, 1,
     integrate.quad(lambda x: 1/(x**2+0.001), 0, 1)[0],
     ["near_singular", "peaked", "rational"]),
 
    # ── Discontinuous / piecewise-smooth ─────────────────────────
    ("abs(sin(x))",    lambda x: np.abs(np.sin(x)),           0, 2*np.pi, 4.0,
     ["piecewise", "non_smooth", "oscillatory"]),
    ("Heaviside at 0.5", lambda x: (x > 0.5).astype(float),  0, 1,  0.5,
     ["piecewise", "discontinuous"]),
    ("x*sign(sin(x))",  lambda x: x*np.sign(np.sin(np.pi*x)+1e-15), 0, 4,
     integrate.quad(lambda x: x*np.sign(np.sin(np.pi*x)), 0, 4)[0],
     ["piecewise", "discontinuous", "polynomial"]),
 
    ("Bessel J0(x)",   lambda x: np.array([float(__import__('scipy').special.j0(xi)) for xi in np.atleast_1d(x)]) if hasattr(x,'__len__') else float(__import__('scipy').special.j0(x)),
     0, 10,
     integrate.quad(lambda x: __import__('scipy').special.j0(x), 0, 10)[0],
     ["special_function", "oscillatory", "decay"]),
]
 
# Convenience: fix exact value for x^4 - 3x^2 + 1 properly
DATASET[1] = (
    "x^4 - 3x^2 + 1",
    lambda x: x**4 - 3*x**2 + 1,
    -1, 2,
    integrate.quad(lambda x: x**4 - 3*x**2 + 1, -1, 2)[0],
    ["polynomial", "smooth", "oscillatory"]
)
 