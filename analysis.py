from dataset import DATASET
from methods import METHODS
def convergence_study(fname="sin(x)", ns=None):
    """Plot absolute error vs n for all methods on one function."""
    if ns is None:
        ns = [8, 16, 32, 64, 128, 256, 512, 1024]
 
    # Find the function entry
    entry = next(e for e in DATASET if e[0] == fname)
    _, f, a, b, exact, _ = entry
 
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = cm.tab10(np.linspace(0, 1, len(METHODS)))
 
    for (mname, mfunc), color in zip(METHODS.items(), colors):
        errors = []
        for n in ns:
            try:
                val = mfunc(f, a, b, n) if mname not in ("Romberg", "Adaptive Simpson") else mfunc(f, a, b)
                errors.append(abs(val - exact))
            except:
                errors.append(np.nan)
        ax.loglog(ns, errors, marker="o", label=mname, color=color, linewidth=1.8)
 
    ax.set_xlabel("Number of points (n)", fontsize=12)
    ax.set_ylabel("Absolute Error", fontsize=12)
    ax.set_title(f"Convergence Study — f(x) = {fname}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("convergence_study.png", dpi=150)
    plt.show()
    print("Saved: convergence_study.png")
 

 
def plot_error_heatmap(df):
    pivot = df.pivot_table(
        index="method", columns="function",
        values="abs_error", aggfunc="mean"
    )
    pivot_log = np.log10(pivot.fillna(1) + 1e-20)
 
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
 
 