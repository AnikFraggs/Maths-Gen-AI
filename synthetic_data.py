
import numpy as np
import pandas as pd
from scipy import integrate
from methods import METHODS
from feature_extractor import extract_features
from tqdm import tqdm

# Define a pool of atomic functions and operations
def safe_log(x):
    return np.log(np.abs(x) + 1e-3)

def safe_div(x):
    return 1.0 / (np.power(x, 2) + 1e-3)

def safe_exp(x):
    # Prevent overflow by clipping extreme values
    return np.exp(np.clip(x, -10, 10))

def generate_random_function():
    """Generates highly complex, nested mathematical functions."""
    a, b = np.random.uniform(-3, 3, 2)
    if a > b: a, b = b, a
    
    # Randomly select interval types (some near 0 for singularities, some wide)
    if np.random.rand() > 0.7:
        a, b = 0, np.random.uniform(0.5, 3.0) # Force endpoint near 0
    
    c1, c2, c3 = np.random.uniform(-2, 2, 3)
    
    # Atomic building blocks
    components = [
        lambda x: x,
        lambda x: x**2,
        lambda x: x**3,
        lambda x: np.sin(c1 * x),
        lambda x: np.cos(c2 * x),
        lambda x: safe_exp(x),
        lambda x: safe_log(x),
        lambda x: safe_div(x),
        lambda x: np.sqrt(np.abs(x)),
        lambda x: 1.0 / (np.sqrt(np.abs(x - c3)) + 1e-3) # Sharp singularity
    ]
    
    # Randomly select 2 or 3 components
    n_comp = np.random.randint(2, 4)
    chosen = np.random.choice(components, n_comp, replace=False)
    
    # Combine them: either multiply, add, or nest them
    op_type = np.random.randint(0, 3)
    
    if op_type == 0: # Product (e.g., sin(x) * exp(x) * log(x))
        f = lambda x: np.prod([c(x) for c in chosen], axis=0)
    elif op_type == 1: # Nested (e.g., sin(exp(log(x))))
        f = lambda x: x
        for c in reversed(chosen):
            f = lambda x, prev_f=f, curr_c=c: curr_c(prev_f(x))
    else: # Sum of products
        f = lambda x: chosen[0](x) + chosen[1](x)
        if n_comp == 3:
            f = lambda x: f(x) * chosen[2](x)
            
    return f, a, b

def calculate_exact(f, a, b):
    """Uses scipy.integrate.quad as the ground truth calculator."""
    try:
        # Increase limit for complex oscillatory functions
        val, _ = integrate.quad(f, a, b, limit=200, epsabs=1e-8, epsrel=1e-8)
        if np.isfinite(val):
            return val
        return np.nan
    except:
        return np.nan

def generate_synthetic_dataset(n_samples=80000):
    print(f"Generating {n_samples} EXTREMELY COMPLEX synthetic functions...")
    data = []
    
    for i in tqdm(range(n_samples)):
        f, a, b = generate_random_function()
        exact = calculate_exact(f, a, b)
        
        # Reject if exact integral fails or is too large (overflow)
        if np.isnan(exact) or abs(exact) > 1e5:
            continue
            
        # Use 150 samples to extract features to catch high-frequency oscillations
        feats = extract_features(f, a, b, samples=150)
        if feats is None:
            continue
            
        row = {"exact": exact}
        row.update(feats)
        
        method_errors = {}
        for mname, mfunc in METHODS.items():
            try:
                # Increase points for complex functions so methods have a fair chance
                if mname in ["Romberg", "Adaptive Simpson"]:
                    val = mfunc(f, a, b)
                elif mname == "Tanh-Sinh":
                    val = mfunc(f, a, b, h=0.05)
                elif mname == "Bayesian QD":
                    val = mfunc(f, a, b, n=25)
                else:
                    val = mfunc(f, a, b, n=150) # Upgraded from 50 to 150
                    
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
    df.to_csv("synthetic_dataset_80k_complex.csv", index=False)
    print(f"✅ Saved {len(df)} valid complex cases to synthetic_dataset_80k_complex.csv")
    return df
