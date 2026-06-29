import numpy as np

def extract_features(f, a, b, samples=200):
    """Analyzes a function and returns mathematical features for the AI."""
    x = np.linspace(a + 1e-9, b - 1e-9, samples)
    try:
        y = np.vectorize(f)(x)
    except:
        return None
    
    finite = np.isfinite(y)
    if finite.sum() < 10: return None
    y = y[finite]; x = x[finite]
    
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    d4y = np.gradient(d2y, x)
    
    sign_changes = np.sum(np.diff(np.sign(dy)) != 0)
    
    return {
        "range_y": float(np.ptp(y)),
        "mean_y": float(np.mean(y)),
        "std_y": float(np.std(y)),
        "max_abs_y": float(np.max(np.abs(y))),
        "max_abs_dy": float(np.max(np.abs(dy))),
        "max_abs_d2y": float(np.max(np.abs(d2y))),
        "max_abs_d4y": float(np.max(np.abs(d4y))),
        "oscillations": float(sign_changes),
        "endpoint_left_mag": float(abs(y[0])),
        "endpoint_right_mag": float(abs(y[-1])),
        "spikiness": float(np.max(np.abs(y)) / (np.mean(np.abs(y)) + 1e-12)),
        "interval_length": float(b - a),
    }
