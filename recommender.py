from dataset import DATASET
from methods import METHODS
def plot_category_breakdown(df):
    # Explode tags to get one row per (function, method, tag)
    exploded = []
    for _, row in df.iterrows():
        for tag in row["tags"].split("|"):
            exploded.append({**row.to_dict(), "tag": tag})
    edf = pd.DataFrame(exploded)
 
    top_tags = edf["tag"].value_counts().head(8).index.tolist()
    edf_top = edf[edf["tag"].isin(top_tags)]
 
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=False)
    axes = axes.flatten()
 
    for i, tag in enumerate(top_tags):
        ax = axes[i]
        sub = edf_top[edf_top["tag"] == tag]
        order_med = (
            sub.groupby("method")["abs_error"].median()
               .reindex(list(METHODS.keys()))
               .fillna(1)
        )
        colors_bar = cm.RdYlGn_r(
            (np.log10(order_med.values + 1e-20) + 14) / 14
        )
        ax.barh(order_med.index, np.log10(order_med.values + 1e-20),
                color=colors_bar)
        ax.axvline(-6, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"Tag: {tag}", fontsize=10, fontweight="bold")
        ax.set_xlabel("log₁₀(median err)", fontsize=8)
 
    plt.suptitle("Method Accuracy per Function Category",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("category_breakdown.png", dpi=150)
    plt.show()
    print("Saved: category_breakdown.png")