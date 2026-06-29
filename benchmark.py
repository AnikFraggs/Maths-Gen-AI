from dataset import DATASET
from methods import METHODS
METHODS = {
    "Trapezoid":         IntegrationMethods.trapezoid,
    "Simpson":           IntegrationMethods.simpson,
    "Boole":             IntegrationMethods.boole,
    "Romberg":           IntegrationMethods.romberg,
    "Gaussian":          IntegrationMethods.gaussian_quadrature,
    "Adaptive Simpson":  IntegrationMethods.adaptive_simpson,
    "Cubic Spline":      IntegrationMethods.cubic_spline,
    "Spectral":          IntegrationMethods.spectral,
    "Bayesian QD":       IntegrationMethods.bayesian_quadrature,
    "Tanh-Sinh":         IntegrationMethods.tanh_sinh,
}
 
def benchmark(n_points=500, verbose=True):
    """
    Runs every method on every function. Returns a DataFrame of results.
    Columns: function, method, absolute_error, relative_error,
             time_ms, converged, tags
    """
    records = []
 
    for (fname, f, a, b, exact, tags) in DATASET:
        if verbose:
            print(f"\n  ▶  {fname}  [{a}, {b}]  exact={exact:.8f}")
 
        for mname, mfunc in METHODS.items():
            t0 = time.perf_counter()
            try:
                result = mfunc(f, a, b)
                ok = True
            except Exception as e:
                result = float("nan")
                ok = False
            elapsed = (time.perf_counter() - t0) * 1000  # ms
 
            abs_err = abs(result - exact) if ok else float("nan")
            rel_err = abs_err / (abs(exact) + 1e-15)
            converged = ok and abs_err < 1e-6
 
            records.append({
                "function":       fname,
                "method":         mname,
                "exact":          exact,
                "computed":       result,
                "abs_error":      abs_err,
                "rel_error":      rel_err,
                "time_ms":        elapsed,
                "converged":      converged,
                "tags":           "|".join(tags),
            })
 
            if verbose:
                status = "✓" if converged else "✗"
                print(f"    {status} {mname:<20}  result={result:.8f}  "
                      f"err={abs_err:.2e}  t={elapsed:.2f}ms")
 
    return pd.DataFrame(records)
 