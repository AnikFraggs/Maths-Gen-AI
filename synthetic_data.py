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
    atoms = [x_sym, x_sym**2, sp.sin(x_sym), sp.cos(x_sym), sp.exp(x_sym), sp.log(x_sym), sp.sqrt(sp.Abs(x_sym))]
    
    expr = np.random.choice(atoms)
    ops = [sp.sin, sp.cos, sp.exp, sp.log, lambda y: y**2, lambda y: sp.sqrt(sp.Abs(y))]
    
    for _ in range(depth):
        op = np.random.choice(ops)
        try:
            expr = op(expr)
        except:
            pass
            
    expr2 = np.random.choice(atoms)
    expr = expr * expr2
    
    if np.random.rand() > 0.6:
        c = np.random.uniform(0.1, 2.0)
        expr = expr / (x_sym**2 + c)
        
    return expr

def calculate_exact_mpmath(f_mp, a, b):
    """Uses mpmath for arbitrary-precision quadrature (The Ultimate Calculator)."""
    try:
        mpmath.mp.dps = 15
        val = mpmath.quad(f_mp, [a, b], maxdegree=9)
        val = float(val)
        if np.isfinite(val) and abs(val) < 1e6:
            return val
        return np.nan
    except:
        return np.nan

def generate_synthetic_dataset(n_samples=80000):
    print(f"Generating {n_samples} EXTREMELY COMPLEX SymPy functions...")
    data = []
    
    for i in tqdm(range(n_samples)):
        # 1. Generate Symbolic Function
        expr = generate_sympy_function(depth=np.random.randint(2, 5))
        
        # 2. Convert to Numpy function (for features) and Mpmath function (for exact calc)
        try:
            f_np = sp.lambdify(x_sym, expr, modules=[SAFE_DICT, "numpy"])
            f_mp = sp.lambdify(x_sym, expr, modules="mpmath")
        except:
            continue
            
        # 3. Generate bounds (FIXED: generating single floats, not arrays)
        if np.random.rand() > 0.7:
            a = 0.0
            b = float(np.random.uniform(0.5, 3.0))
        else:
            a = float(np.random.uniform(-3, 0))
            b = float(np.random.uniform(0, 3))
            if a > b: 
                a, b = b, a
            
        # 4. Calculate EXACT calculator value using mpmath
        exact = calculate_exact_mpmath(f_mp, a, b)
        if np.isnan(exact):
            continue
            
        # 5. Extract Advanced Features (FFT & High Derivatives)
        try:
            samples = 256
            xx = np.linspace(a + 1e-9, b - 1e-9, samples)
            yy = np.vectorize(f_np)(xx)
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
                    val = mfunc(f_np, a, b)
                elif mname == "Tanh-Sinh":
                    val = mfunc(f_np, a, b, h=0.02) # Ultra-fine step
                elif mname == "Bayesian QD":
                    val = mfunc(f_np, a, b, n=50)   # More GP points
                else:
                    val = mfunc(f_np, a, b, n=2000) # Max resolution for grid methods
                    
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
    print(f"✅ Saved {len(df)} ULTRA-COMPLEX cases to synthetic_dataset_100k_ultimate.csv")
    return df
