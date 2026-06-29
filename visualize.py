from dataset import DATASET
from methods import METHODS
def empirical_order(f, a, b, exact, mfunc, ns=(100, 200, 400, 800)):
    """
    Estimate convergence order p from error ~ C*n^(-p).
    Returns mean of log(e1/e2)/log(n2/n1) over consecutive pairs.
    """
    errors = []
    for n in ns:
        try:
            e = abs(mfunc(f, a, b, n) - exact)
        except:
            e = float("nan")
        errors.append(e)
    orders = []
    for i in range(len(ns)-1):
        if errors[i] > 1e-14 and errors[i+1] > 1e-14:
            orders.append(np.log(errors[i]/errors[i+1]) / np.log(ns[i+1]/ns[i]))
    return np.nanmean(orders) if orders else float("nan")
 
 
def compute_orders(df):
    print("\n<── <<<<<<<<<<Empirical convergence orders >>>>>>>─────────────────>")
    rows = []
    for mname, mfunc in METHODS.items():
        if mname in ("Romberg", "Adaptive Simpson", "Bayesian QD", "Tanh-Sinh"):
            continue
        for (fname, f, a, b, exact, tags) in DATASET[:10]:   # first 10
            o = empirical_order(f, a, b, exact, mfunc)
            rows.append({"method": mname, "function": fname, "order": round(o, 2)})
    odf = pd.DataFrame(rows)
    pivot = odf.pivot_table(index="method", columns="function",
                             values="order", aggfunc="mean")
    print(tabulate(pivot.fillna("-"), headers="keys", tablefmt="rounded_grid",
                   floatfmt=".2f"))
    return odf
 
 
def build_pattern_model(df):
    """
    For each (tag combination → method), find which method has
    the lowest median relative error. Returns a dict and prints
    a ranked table per tag.
    """
    # Expand tags
    rows = []
    for _, row in df.iterrows():
        for tag in row["tags"].split("|"):
            rows.append({
                "tag":       tag,
                "method":    row["method"],
                "rel_error": row["rel_error"],
                "converged": row["converged"],
            })
    tdf = pd.DataFrame(rows)
 
    # For each tag: median relative error per method
    perf = (
        tdf.groupby(["tag", "method"])["rel_error"]
           .median()
           .reset_index()
           .rename(columns={"rel_error": "median_rel_err"})
    )
 
    # Best method per tag
    best = perf.loc[perf.groupby("tag")["median_rel_err"].idxmin()]
 
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║         PATTERN MODEL — Best Method per Tag         ║")
    print("╠══════════════════════════════════════════════════════╣")
 
    # Full ranked table per tag
    all_tags = sorted(tdf["tag"].unique())
    pattern_dict = {}
    for tag in all_tags:
        sub = perf[perf["tag"] == tag].sort_values("median_rel_err")
        best_m = sub.iloc[0]["method"]
        best_e = sub.iloc[0]["median_rel_err"]
        pattern_dict[tag] = best_m
        print(f"\n  TAG: {tag}")
        for _, r in sub.iterrows():
            bar = "█" * max(1, int(20 - np.clip(-np.log10(r["median_rel_err"]+1e-20), 0, 20)))
            marker = " ◄ BEST" if r["method"] == best_m else ""
            print(f"    {r['method']:<22} {r['median_rel_err']:.2e}  {bar}{marker}")
 
    print("\n╚══════════════════════════════════════════════════════╝")
    return pattern_dict
 
 
def recommend(tags, pattern_dict):
    """
    Given a list of tags describing an integrand, recommend
    the top-3 methods by voting.
    """
    votes = {}
    for t in tags:
        m = pattern_dict.get(t)
        if m:
            votes[m] = votes.get(m, 0) + 1
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    print("\n┌─ RECOMMENDATION ────────────────────────────────────┐")
    for i, (method, v) in enumerate(ranked[:3], 1):
        print(f"│  #{i}  {method:<22}  (matched {v} tag(s))")
    print("└─────────────────────────────────────────────────────┘")
    return ranked
 
def summary_table(df):
    summary = df.groupby("method").agg(
        mean_abs_error=("abs_error", "mean"),
        median_abs_error=("abs_error", "median"),
        convergence_rate=("converged", "mean"),
        mean_time_ms=("time_ms", "mean"),
    ).sort_values("median_abs_error")
 
    summary["convergence_%"] = (summary["convergence_rate"] * 100).map("{:.1f}%".format)
    summary = summary.drop(columns=["convergence_rate"])
 
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║                  METHOD SUMMARY (all functions)                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(tabulate(summary, headers="keys", tablefmt="rounded_grid", floatfmt=".4e"))
    return summary