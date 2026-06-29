from dataset import DATASET
from methods import METHODS
def run_all():
    print("="*65)
    print("  NUMERICAL INTEGRATION BENCHMARKING — FULL PIPELINE")
    print("="*65)
    print(f"\n  Functions  : {len(DATASET)}")
    print(f"  Methods    : {len(METHODS)}")
    print(f"  Total runs : {len(DATASET) * len(METHODS)}\n")
 
    # ── Step 1: Benchmark ─────────────────────────────────────────
    print("─"*65)
    print("  STEP 1 / 6  Benchmarking all methods × all functions …")
    print("─"*65)
    df = benchmark(verbose=True)
    df.to_csv("integration_results.csv", index=False)
    print("\n  Results saved → integration_results.csv")
 
    # ── Step 2: Summary table ─────────────────────────────────────
    print("\n" + "─"*65)
    print("  STEP 2 / 6  Summary table")
    print("─"*65)
    summary_table(df)
 
    # ── Step 3: Convergence orders ────────────────────────────────
    print("\n" + "─"*65)
    print("  STEP 3 / 6  Empirical convergence orders")
    print("─"*65)
    compute_orders(df)
 
    # ── Step 4: Pattern model ─────────────────────────────────────
    print("\n" + "─"*65)
    print("  STEP 4 / 6  Pattern recognition model")
    print("─"*65)
    pattern_dict = build_pattern_model(df)
 
    # ── Step 5: Plots ─────────────────────────────────────────────
    print("\n" + "─"*65)
    print("  STEP 5 / 6  Generating plots …")
    print("─"*65)
    plot_dashboard(df)
    plot_error_heatmap(df)
    plot_category_breakdown(df)
    convergence_study("sin(x)")
    convergence_study("e^(-x^2)")
 
    # ── Step 6: Demo recommendation ──────────────────────────────
    print("\n" + "─"*65)
    print("  STEP 6 / 6  Demo — Recommend method for a new integral")
    print("─"*65)
    print("\n  Query: 'I have a highly oscillatory, smooth function'")
    recommend(["oscillatory", "smooth", "high_frequency"], pattern_dict)
 
    print("\n  Query: 'My function has an endpoint singularity'")
    recommend(["endpoint_singularity", "singular"], pattern_dict)
 
    print("\n  Query: 'Sampled experimental data, smooth curve'")
    recommend(["smooth", "no_singularity"], pattern_dict)
 
    print("\n  Query: 'Peaked Gaussian-like function'")
    recommend(["peaked", "gaussian_shape"], pattern_dict)
 
    print("\n" + "="*65)
    print("  ALL DONE.  Check integration_results.csv + *.png files.")
    print("="*65)
    return df, pattern_dict
 
 