
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from synthetic_data import generate_synthetic_dataset

def run_neural_pipeline():
    # 1. Generate or Load Data
    try:
        df = pd.read_csv("synthetic_dataset_80k_ultimate.csv")
        print("✅ Loaded existing synthetic_dataset_80k_ultimate.csv")
    except FileNotFoundError:
        df = generate_synthetic_dataset(n_samples=100000)
        
    # 2. Prepare Data for Neural Network (Added new features)
    feature_cols = [
        "range_y", "mean_y", "std_y", "max_abs_y", "max_abs_dy",
        "max_abs_d2y", "max_abs_d4y", "max_abs_d6y", "oscillations",
        "endpoint_left_mag", "endpoint_right_mag", "spikiness", "interval_length",
        "dominant_frequency", "spectral_entropy"
    ]
    
    df = df.dropna(subset=feature_cols + ["best_method"])
    X = df[feature_cols].values
    y = df["best_method"].values
    
    indices = np.arange(len(df))
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Build and Train Massive Deep Neural Network (150 epochs)
    print("\n🧠 Training Massive Deep Neural Network (150 epochs) on Ultra-Complex Functions...")
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64, 32),  # 5-Layer Network
        activation='relu',
        solver='adam',
        max_iter=150,          # 150 Epochs
        verbose=True,
        random_state=42,
        learning_rate_init=0.0005,
        early_stopping=False,
        batch_size=512         # Huge batch for deep learning stability
    )
    
    clf.fit(X_train_scaled, y_train)
    
    # 4. Plot Epoch vs Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(clf.loss_curve_, color='crimson', linewidth=2.5)
    plt.title("Deep Neural Network Training Loss (Ultra-Complex Functions)", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss (Cross-Entropy)", fontsize=12)
    plt.yscale('log') # Log scale to see tiny loss improvements
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("epoch_loss_curve_ultimate.png", dpi=150)
    plt.show()
    
    # 5. Evaluate AI on Test Set
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 AI Accuracy on Test Set: {acc:.2%}")
    
    # 6. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=clf.classes_, 
                yticklabels=clf.classes_, cmap="Blues")
    plt.title("AI Confusion Matrix (Ultra-Complex)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("confusion_matrix_ultimate.png", dpi=150)
    plt.show()
    
    # 7. Calculate AI Loss vs Original Calculator (Mpmath)
    print("\nCalculating AI method loss vs Original Calculator (Mpmath)...")
    test_df = df.iloc[test_idx].copy()
    test_df["ai_predicted_method"] = y_pred
    
    ai_errors = []
    calculator_errors = []
    
    for _, row in test_df.iterrows():
        ai_method = row["ai_predicted_method"]
        ai_err = row[ai_method] if not np.isnan(row[ai_method]) else 1.0
        calc_err = row["best_error"] if not np.isnan(row["best_error"]) else 1.0
        
        ai_errors.append(min(ai_err, 1.0))
        calculator_errors.append(min(calc_err, 1.0))
        
    # 8. Plot the Loss Comparison Graph (Advanced Hexbin)
    plt.figure(figsize=(12, 9))
    
    # Log scale transform for plotting
    log_calc = np.log10(np.array(calculator_errors) + 1e-16)
    log_ai = np.log10(np.array(ai_errors) + 1e-16)
    
    hb = plt.hexbin(
        log_calc, log_ai, 
        gridsize=60, 
        cmap='magma', 
        mincnt=1,
        alpha=0.9
    )
    
    lims = [-16, 0]
    plt.plot(lims, lims, 'c--', linewidth=3, label="Perfect Prediction (y=x)")
    
    plt.colorbar(hb, label='Density of 16,000 Test Functions')
    plt.title("AI Integration Loss vs Original Calculator Loss\n(Ultra-Complex Nested Composites)", fontsize=14, fontweight='bold')
    plt.xlabel("log₁₀(Calculator Error) [Mpmath Ground Truth]", fontsize=12)
    plt.ylabel("log₁₀(AI Error) [AI Selected Method]", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("ai_vs_calculator_loss_ultimate.png", dpi=150)
    plt.show()
    print("✅ Saved ultimate evaluation graphs!")
import joblib
joblib.dump(clf, 'ai_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Saved AI model to ai_model.pkl and scaler.pkl for web app use!")


if __name__ == "__main__":
    run_neural_pipeline()
