import time
import pandas as pd
import numpy as np
from methods import METHODS
from dataset import DATASET
from feature_extractor import extract_features

def benchmark():
    records = []
    print("Running benchmarks...")
    
    for fname, f, a, b, exact, tags in DATASET:
        # Extract features for AI
        feats = extract_features(f, a, b) or {}
        
        # Find best method for this function
        temp_results = {}
        for mname, mfunc in METHODS.items():
            try:
                val = mfunc(f, a, b)
                err = abs(val - exact) / (abs(exact) + 1e-15)
                temp_results[mname] = err
            except:
                temp_results[mname] = np.nan
        
        best_method = pd.Series(temp_results).idxmin()
        
        for mname, mfunc in METHODS.items():
            t0 = time.perf_counter()
            try:
                result = mfunc(f, a, b)
                ok = True
            except:
                result = float("nan"); ok = False
            elapsed = (time.perf_counter() - t0) * 1000
            
            abs_err = abs(result - exact) if ok else float("nan")
            
            row = {
                "function": fname, "method": mname, "abs_error": abs_err,
                "time_ms": elapsed, "converged": ok and abs_err < 1e-6,
                "best_method": best_method
            }
            row.update(feats) # Add AI features to every row
            records.append(row)
            
    return pd.DataFrame(records)
