from dataset import DATASET
from methods import METHODS
def plot_dashboard(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Numerical Integration Benchmarking Dashboard",
                 fontsize=15, fontweight="bold", y=1.01)
 
    methods = list(METHODS.keys())
    colors  = cm.tab10(np.linspace(0, 1, len(methods)))
 
    # ── Panel A: Median absolute error per method ─────────────────
    ax = axes[0, 0]
    med_err = df.groupby("method")["abs_error"].median().reindex(methods)
    ax.barh(methods, np.log10(med_err + 1e-20), color=colors)
    ax.axvline(-6, color="red", linestyle="--", linewidth=1, label="1e-6 threshold")
    ax.set_xlabel("log₁₀(median absolute error)")
    ax.set_title("A — Accuracy (lower = better)")
    ax.legend(fontsize=8)
 
    # ── Panel B: Mean computation time ───────────────────────────
    ax = axes[0, 1]
    mean_time = df.groupby("method")["time_ms"].mean().reindex(methods)
    ax.barh(methods, mean_time, color=colors)
    ax.set_xlabel("Mean time (ms)")
    ax.set_title("B — Speed (lower = better)")
 
    # ── Panel C: Convergence rate (% of functions converged) ─────
    ax = axes[1, 0]
    conv = df.groupby("method")["converged"].mean().reindex(methods) * 100
    ax.barh(methods, conv, color=colors)
    ax.axvline(90, color="green", linestyle="--", linewidth=1, label="90%")
    ax.set_xlabel("% functions converged (err < 1e-6)")
    ax.set_title("C — Robustness")
    ax.set_xlim(0, 105)
    ax.legend(fontsize=8)
 
    # ── Panel D: Scatter accuracy vs speed ───────────────────────
    ax = axes[1, 1]
    for (mname, _), color in zip(METHODS.items(), colors):
        sub = df[df["method"] == mname]
        ax.scatter(
            sub["time_ms"].mean(),
            np.log10(sub["abs_error"].median() + 1e-20),
            color=color, s=120, label=mname, zorder=5
        )
        ax.annotate(mname, (sub["time_ms"].mean(),
                             np.log10(sub["abs_error"].median() + 1e-20)),
                    fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Mean time (ms)")
    ax.set_ylabel("log₁₀(median abs error)")
    ax.set_title("D — Accuracy–Speed Tradeoff")
    ax.axhline(-6, color="red", linestyle="--", linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: dashboard.png")
 