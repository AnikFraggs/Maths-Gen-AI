import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataset import DATASET
from methods import METHODS

def convergence_study(fname="sin(x)", ns=None):
    """Plot absolute error vs n for all methods on one function."""
    if ns is None:
        ns = [8, 16, 32, 64, 128, 256, 512]
 
    # Find the function entry safely
    entry = next((e for e in DATASET if e[0] == fname), None)
    if entry is None:
        print(f"Function '{fname}' not found in dataset.")
        return
        
    _, f, a, b, exact, _ = entry
 
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = cm.tab10(np.linspace(0, 1, len(METHODS)))
 
    for (mname, mfunc), color in zip(METHODS.items(), colors):
        errors = []
        for n in ns:
            try:
                # Pass 'n' only if the method accepts it.
                # For Tanh-Sinh and Romberg, we call them without 'n' 
                # because they handle step sizes differently.
                if mname in ["Romberg", "Adaptive Simpson"]:
                    val = mfunc(f, a, b)
                elif mname == "Tanh-Sinh":
                    # Tanh-sinh uses 'h' (step size) instead of 'n' directly.
                    # We map n to a step size to simulate increasing resolution.
                    h = 10.0 / n
                    val = mfunc(f, a, b, h=h)
                else:
                    val = mfunc(f, a, b, n)
                    
                err = abs(val - exact)
                # Prevent log(0) errors by setting a floor
                errors.append(max(err, 1e-16))
            except Exception as e:
                errors.append(np.nan)
                
        # Only plot if we have valid data
        valid_errors = [e for e in errors if not np.isnan(e)]
        if valid_errors:
            ax.loglog(ns[:len(valid_errors)], valid_errors, marker="o", label=mname, color=color, linewidth=1.8)
 
    ax.set_xlabel("Number of points (n) / Resolution", fontsize=12)
    ax.set_ylabel("Absolute Error", fontsize=12)
    ax.set_title(f"Convergence Study — f(x) = {fname}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("convergence_study.png", dpi=150)
    plt.show()
    print("Saved: convergence_study.png")
 

def plot_error_heatmap(df):
    """Plots a heatmap of log(error) for methods vs functions."""
    pivot = df.pivot_table(
        index="method", columns="function",
        values="abs_error", aggfunc="mean"
    )
    # Replace 0 or NaN with tiny numbers so log10 doesn't break
    pivot_log = np.log10(pivot.fillna(1) + 1e-16)
 
    fig, ax = plt.subplots(figsize=(22, 7))
    im = ax.imshow(pivot_log.values, aspect="auto", cmap="RdYlGn_r",
                   vmin=-14, vmax=0)
 
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_title("log₁₀(|Error|) Heatmap — Methods × Functions\n"
                 "(greener = more accurate)", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="log₁₀(absolute error)")
    plt.tight_layout()
    plt.savefig("heatmap_error.png", dpi=150)
    plt.show()
    print("Saved: heatmap_error.png")
