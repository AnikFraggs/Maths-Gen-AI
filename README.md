🧮 AI-Guided Numerical Integration Selector
This project uses a Deep Neural Network (MLP) to predict the most accurate numerical integration method for any given mathematical function. It benchmarks 10 distinct integration algorithms against an arbitrary-precision calculator (mpmath) to learn complex mathematical patterns.
<img width="1895" height="99" alt="image" src="https://github.com/user-attachments/assets/5460baa3-1247-4da5-844e-bb21f14c8b37" />
<img width="1895" height="99" alt="image" src="https://github.com/user-attachments/assets/c00430ea-7f61-48a0-8e9c-cac4846203d1" />

📈 Project Evolution & Results
Phase 1: Standard Functions (85.43% Accuracy)
Initially, the AI was trained on 80,000 standard functions (basic polynomials, simple trig, standard exponentials). Because the mathematical patterns were clear and distinct, the AI easily achieved 85.43% accuracy. A ConvergenceWarning was observed but the model classified smooth vs. singular functions perfectly.

Phase 2: Ultra-Complex Nested Functions (44.57% Accuracy)
To test the limits of the AI, the dataset generator was upgraded to use sympy to create deeply nested mathematical monstrosities (e.g., exp(sin(log(x^2))) / (x^2 + c)).
->Why did accuracy drop to 44.57%? At this level of complexity, many numerical methods fail completely (returning NaN or division-by-zero). The "best" method often wins simply by surviving the calculation, making the data highly noisy and contradictory.
->44% accuracy on functions this complex is actually a testament to the AI's ability to find patterns in mathematical chaos.

Phase 3: Filtered Ultra-Complex (Optimized)
By applying data quality filters (removing impossible-to-integrate pathological functions) and adding L2 Regularization to the Neural Network, the model stabilized its predictions on highly oscillatory and composite functions.


🚀 The 10 Integration Methods
1>Trapezoid
2>Simpson
3>Boole
4>Romberg
5>Gaussian Quadrature
6>Adaptive Simpson
7>Cubic Spline
8>Spectral (Clenshaw-Curtis)
9>Bayesian Quadrature (GP-based)
10>Tanh-Sinh (Double Exponential)


💻 How to Run
1. Train the Model:pip install numpy scipy pandas scikit-learn matplotlib seaborn tabulate tqdm sympy mpmath flask
Generate data and train the AI:
python train_neural.py
2. Launch the Web Application
Once the model is trained and saved, start the Flask server:python app.py

As this a local run project on devie use local host :Open your browser and go to http://127.0.0.1:5000. Input any function in terms of x, set your limits, and let the AI recommend the best integration method!
