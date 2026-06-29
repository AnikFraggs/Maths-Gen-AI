import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from feature_extractor import extract_features

def train_ai_model(df):
    """Trains a Random Forest to predict the best integration method."""
    features_list = []
    best_methods = []
    
    for _, row in df.iterrows():
        # We need the original function. Since we only have 'function' name in df,
        # in a real scenario you'd pass the function. For this script, we'll mock 
        # feature extraction by pulling from a pre-calculated column if it exists.
        # To make this work seamlessly, we update df in benchmark.py
        pass 
    
    # This function assumes df already has feature columns attached
    feature_cols = ["range_y", "mean_y", "std_y", "max_abs_y", "max_abs_dy",
                    "max_abs_d2y", "max_abs_d4y", "oscillations",
                    "endpoint_left_mag", "endpoint_right_mag", "spikiness", "interval_length"]
    
    X = df[feature_cols].fillna(0).values
    y = df["best_method"].values
    
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    clf.fit(Xtr, ytr)
    
    print("\n🤖 AI Model Trained!")
    print(f"Accuracy on test set: {clf.score(Xte, yte):.2%}")
    print("\nClassification Report:")
    print(classification_report(yte, clf.predict(Xte), zero_division=0))
    
    return clf, feature_cols

def predict_best_method(clf, feature_cols, f, a, b):
    """Uses the trained AI to recommend a method for a new function."""
    feats = extract_features(f, a, b)
    if feats is None: return "Unknown"
    
    x_in = np.array([feats[c] for c in feature_cols]).reshape(1, -1)
    return clf.predict(x_in)[0]
