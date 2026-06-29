# synthetic_data.py
import numpy as np
import pandas as pd
from scipy import integrate
import sympy as sp
import mpmath
from methods import METHODS
from tqdm import tqdm

# Define symbolic variables and safe numpy mappings
x_sym = sp.symbols('x')
SAFE_DICT = {
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'exp': np.exp,
    'log': np.log, 'sqrt': np.sqrt, 'Abs': np.abs, 'pi': np.pi, 'E': np.e
}

def generate_sympy_function(depth=3):
    """Generates deeply nested mathematical functions using SymPy."""
    # Base atoms
    atoms = [x_sym, x_sym**2, sp.sin(x_sym), sp.cos(x_sym), sp.exp(x_sym), sp.log(x_sym), sp.sqrt(sp.Abs(x_sym))]
    
    expr = np.random.choice(atoms)
    ops = [sp.sin, sp.cos, sp.exp, sp.log, lambda y: y**2, lambda y: sp.sqrt(sp.Abs(y))]
    
    # Nest operations deeply
    for _ in range(depth):
        op = np.random.choice(ops)
        try:
            expr = op(expr)
        except:
            pass
            
    # Multiply by another random base function to create composites
    expr2 = np.random.choice(atoms)
    expr = expr * expr2
    
    # Add a rational / singular component sometimes
    if np.random.rand() > 0.6:
        c = np.random.uniform(0.1, 2.0)
        expr = expr / (x_sym**2 + c)
        
    return expr

def calculate_exact_mpmath(f_lambda, a, b):
    """Uses mpmath for arbitrary-precision quadrature (The Ultimate Calculator)."""
    try:
        # 15 decimal places of precision
        mpmath.mp.dps = 15
        val = mpmath.quad(f_lambda, [a, b], error=True, maxdegree=9)
        if np.isfinite(float(val[0])) and abs(float(val[0])) < 1e6:
            return float(val[0])
        return np.nan
    except:
        return np.nan

def generate_synthetic_dataset(n_samples=80000):
    print(f"Generating {n_samples} EXTREMELY COMPLEX SymPy functions...")
    data = []
    
    for i in tqdm(range(n_samples)):
        # 1. Generate Symbolic Function
        expr = generate_sympy_function(depth=np.random.randint(2, 5))
        
        # 2. Convert to Numpy function
        try:
            f = sp.lambdify(x_sym, expr, modules=[SAFE_DICT, "numpy"])
        except:
            continue
            
        # 3. Generate bounds (some wide, some near 0 for singularities)
        if np.random.rand() > 0.7:
            a, b = 0.0, np.random.uniform(0.5, 3.0)
        else:
            a, b = np.random.uniform(-3, 0, 2), np.random.uniform(0, 3, 2)
            if a > b: a, b = b, a
            
        # 4. Calculate EXACT calculator value using mpmath
        exact = calculate_exact_mpmath(f, a, b)
        if np.isnan(exact):
            continue
            
        # 5. Extract Advanced Features (FFT & High Derivatives)
        try:
            samples = 256
            xx = np.linspace(a + 1e-9, b - 1e-9, samples)
            yy = np.vectorize(f)(xx)
            yy = np.nan_to_num(yy, nan=0.0, posinf=1e5, neginf=-1e5)
            
            # FFT to find dominant frequency
            fft_vals = np.abs(np.fft.fft(yy))
            dominant_freq = np.argmax(fft_vals[1:]) + 1
            
            dy = np.gradient(yy, xx)
            d2y = np.gradient(dy, xx)
            d4y = np.gradient(d2y, xx)
            d6y = np.gradient(d4y, xx)
            
            feats = {
                "range_y": float(np.ptp(yy)),
                "mean_y": float(np.mean(yy)),
                "std_y": float(np.std(yy)),
                "max_abs_y": float(np.max(np.abs(yy))),
                "max_abs_dy": float(np.max(np.abs(dy))),
                "max_abs_d2y": float(np.max(np.abs(d2y))),
                "max_abs_d4y": float(np.max(np.abs(d4y))),
                "max_abs_d6y": float(np.max(np.abs(d6y))),
                "oscillations": float(np.sum(np.diff(np.sign(dy)) != 0)),
                "endpoint_left_mag": float(abs(yy[0])),
                "endpoint_right_mag": float(abs(yy[-1])),
                "spikiness": float(np.max(np.abs(yy)) / (np.mean(np.abs(yy)) + 1e-12)),
                "interval_length": float(b - a),
                "dominant_frequency": float(dominant_freq),
                "spectral_entropy": float(-np.sum((fft_vals/np.sum(fft_vals)) * np.log(fft_vals/np.sum(fft_vals) + 1e-12))),
            }
        except:
            continue
            
        row = {"exact": exact}
        row.update(feats)
        
        # 6. Run all 10 methods with VERY HIGH resolution (n=2000)
        method_errors = {}
        for mname, mfunc in METHODS.items():
            try:
                if mname in ["Romberg", "Adaptive Simpson"]:
                    val = mfunc(f, a, b)
                elif mname == "Tanh-Sinh":
                    val = mfunc(f, a, b, h=0.02) # Ultra-fine step
                elif mname == "Bayesian QD":
                    val = mfunc(f, a, b, n=50)   # More GP points
                else:
                    val = mfunc(f, a, b, n=2000) # Max resolution for grid methods
                    
                err = abs(val - exact) / (abs(exact) + 1e-12)
                method_errors[mname] = err
            except:
                method_errors[mname] = np.nan
                
        valid_errs = {k: v for k, v in method_errors.items() if not np.isnan(v)}
        if not valid_errs:
            continue
            
        best_m = min(valid_errs, key=valid_errs.get)
        row["best_method"] = best_m
        row["best_error"] = valid_errs[best_m]
        row.update(method_errors)
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv("synthetic_dataset_80k_ultimate.csv", index=False)
    print(f"✅ Saved {len(df)} ULTRA-COMPLEX cases to synthetic_dataset_80k_ultimate.csv")
    return df
