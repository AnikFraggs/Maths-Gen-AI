
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from synthetic_data import generate_synthetic_dataset

def run_neural_pipeline():
    # 1. Generate or Load Data
    try:
        df = pd.read_csv("synthetic_dataset_80k_complex.csv")
        print("✅ Loaded existing synthetic_dataset_80k_complex.csv")
    except FileNotFoundError:
        df = generate_synthetic_dataset(n_samples=100000)
        
    # 2. Prepare Data for Neural Network
    feature_cols = ["range_y", "mean_y", "std_y", "max_abs_y", "max_abs_dy",
                    "max_abs_d2y", "max_abs_d4y", "oscillations",
                    "endpoint_left_mag", "endpoint_right_mag", "spikiness", "interval_length"]
    
    df = df.dropna(subset=feature_cols + ["best_method"])
    X = df[feature_cols].values
    y = df["best_method"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features (Crucial for Neural Networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Build and Train Deep Neural Network (100 epochs)
    print("\n🧠 Training Deep Neural Network (100 epochs) on Complex Functions...")
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),  # Much deeper network
        activation='relu',
        solver='adam',
        max_iter=100,          # 100 Epochs
        verbose=True,
        random_state=42,
        learning_rate_init=0.001,
        early_stopping=False,  # Force it to train all 100 epochs
        batch_size=256         # Larger batch for faster GPU/CPU matrix math
    )
    
    clf.fit(X_train_scaled, y_train)
    
    # 4. Plot Epoch vs Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(clf.loss_curve_, color='red', linewidth=2.5)
    plt.title("Deep Neural Network Training Loss (Complex Functions)", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss (Cross-Entropy)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("epoch_loss_curve_complex.png", dpi=150)
    plt.show()
    
    # 5. Evaluate AI on Test Set
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 AI Accuracy on Test Set: {acc:.2%}")
    
    # 6. Calculate AI Loss vs Original Calculator
    print("\nCalculating AI method loss vs original calculator...")
    # Re-align the test set dataframe
    test_indices = y_test.index
    test_df = df.loc[test_indices].copy()
    test_df["ai_predicted_method"] = y_pred
    
    ai_errors = []
    calculator_errors = []
    
    for _, row in test_df.iterrows():
        ai_method = row["ai_predicted_method"]
        # If AI's method failed (nan), assign a high error penalty
        ai_err = row[ai_method] if not np.isnan(row[ai_method]) else 1.0
        calc_err = row["best_error"] if not np.isnan(row["best_error"]) else 1.0
        
        # Clip errors to 1.0 (100% error) for visualization sanity
        ai_errors.append(min(ai_err, 1.0))
        calculator_errors.append(min(calc_err, 1.0))
        
    # 7. Plot the Loss Comparison Graph (Hexbin Density Plot)
    plt.figure(figsize=(10, 8))
    
    # Use hexbin because 100,000 points overlapping as dots is unreadable
    hb = plt.hexbin(
        np.log10(np.array(calculator_errors) + 1e-16), 
        np.log10(np.array(ai_errors) + 1e-16), 
        gridsize=50, 
        cmap='inferno', 
        mincnt=1,
        alpha=0.8
    )
    
    # Plot y=x line (Perfect prediction line)
    lims = [-16, 0]
    plt.plot(lims, lims, 'c--', linewidth=3, label="Perfect Prediction (y=x)")
    
    plt.colorbar(hb, label='Number of Functions (Density)')
    plt.title("AI Integration Loss vs Original Calculator Loss\n(Complex Composite Functions)", fontsize=14, fontweight='bold')
    plt.xlabel("log₁₀(Calculator Error) [Ground Truth Best Method]", fontsize=12)
    plt.ylabel("log₁₀(AI Error) [AI Selected Method]", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("ai_vs_calculator_loss_complex.png", dpi=150)
    plt.show()
    print("✅ Saved: epoch_loss_curve_complex.png and ai_vs_calculator_loss_complex.png")

if __name__ == "__main__":
    run_neural_pipeline()
