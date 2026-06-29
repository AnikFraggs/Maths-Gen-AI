import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_dashboard(df, methods):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Numerical Integration Benchmark Dashboard", fontsize=14, fontweight="bold")
    
    med_err = df.groupby("method")["abs_error"].median().reindex(methods)
    axes[0].barh(med_err.index, np.log10(med_err + 1e-20), color='skyblue')
    axes[0].set_title("Median Error (log10)")
    
    mean_time = df.groupby("method")["time_ms"].mean().reindex(methods)
    axes[1].barh(mean_time.index, mean_time, color='lightgreen')
    axes[1].set_title("Mean Time (ms)")
    
    conv = df.groupby("method")["converged"].mean().reindex(methods) * 100
    axes[2].barh(conv.index, conv, color='salmon')
    axes[2].set_title("Robustness (% Converged)")
    
    plt.tight_layout()
    plt.savefig("dashboard.png", dpi=150)
    plt.show()

def plot_ai_model_metrics(clf, X_test, y_test, feature_cols):
    """Visualizes AI model performance"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, clf.predict(X_test), labels=clf.classes_)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=clf.classes_, 
                yticklabels=clf.classes_, cmap="Blues", ax=axes[0])
    axes[0].set_title("AI Confusion Matrix")
    axes[0].set_xlabel("Predicted Method")
    axes[0].set_ylabel("Actual Best Method")
    
    # Feature Importance
    imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values()
    imp.plot(kind="barh", ax=axes[1], color='purple')
    axes[1].set_title("AI Feature Importance (What it learned)")
    
    plt.tight_layout()
    plt.savefig("ai_metrics.png", dpi=150)
    plt.show()
