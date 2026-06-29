import pandas as pd
import numpy as np
from benchmark import benchmark
from ai_model import train_ai_model
from visualize import plot_dashboard, plot_ai_model_metrics
from analysis import convergence_study, plot_error_heatmap
from methods import METHODS
from sklearn.model_selection import train_test_split

def main():
    print("="*60)
    print("  AI-GUIDED NUMERICAL INTEGRATION SELECTOR")
    print("="*60)
    
    # 1. Benchmark & Extract Features
    df = benchmark()
    df.to_csv("results.csv", index=False)
    print("✅ Benchmarks complete. Results saved to results.csv")
    
    # 2. Visualize Method Performance
    method_names = list(METHODS.keys())
    plot_dashboard(df, method_names)
    
    # 3. Train AI Model
    print("\n🧠 Training AI to recognize function patterns...")
    feature_cols = ["range_y", "mean_y", "std_y", "max_abs_y", "max_abs_dy",
                    "max_abs_d2y", "max_abs_d4y", "oscillations",
                    "endpoint_left_mag", "endpoint_right_mag", "spikiness", "interval_length"]
    
    X = df[feature_cols].fillna(0).values
    y = df["best_method"].values
    
    # Check if we have enough data to split
    if len(df) >= 5:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Use the train_ai_model function to get the classifier
        clf, _ = train_ai_model(df) 
        clf.fit(Xtr, ytr)
        
        # 4. Visualize AI Performance
        print("\n📊 Generating AI visualizations...")
        plot_ai_model_metrics(clf, Xte, yte, feature_cols)
        
        # 5. Demo AI Prediction
        print("\n🎯 Demo: Asking AI to recommend a method for a sharp spike function:")
        from feature_extractor import extract_features
        test_f = lambda x: 1/(1+1000*(x-0.7)**2)
        feats = extract_features(test_f, 0, 1)
        x_in = np.array([feats[c] for c in feature_cols]).reshape(1, -1)
        prediction = clf.predict(x_in)[0]
        print(f"   Function: Sharp spike at x=0.7")
        print(f"   AI Recommendation: {prediction}")

        # 6. Generate Deep Analysis Graphs
        print("\n📈 Generating Convergence and Heatmap analysis...")
        # Plot heatmap (Uses df)
        plot_error_heatmap(df)
        
        # Plot convergence for a smooth function
        convergence_study("sin(x)")
        
        # Plot convergence for a spiky function
        convergence_study("exp(-100*(x-0.5)^2)")
        
    else:
        print("Not enough data to train AI model.")

if __name__ == "__main__":
    main()
