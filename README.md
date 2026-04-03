# 🌿 Carob Molasses Extraction Predictive Tool

This repository contains a machine learning-based decision support system for optimizing the extraction process of Carob Molasses. 

### 🔬 Research Background
The tool uses **Support Vector Regression (SVR)** with a **Radial Basis Function (RBF) kernel**, optimized via **Bayesian Optimization (Optuna)**. To capture complex chemical behaviors.

### 📊 Performance
The models were validated using **Leave-One-Out Cross-Validation (LOOCV)**, achieving high predictive accuracy:
* **Yield**: $R^2 > 0.95$
* **TPC**: $R^2 > 0.94$
* **5-HMF**: $R^2 > 0.92$

### 🛠️ Features
- **Real-time Prediction**: Adjust extraction time, microwave power, and liquid-to-solid ratio.
- **Optimized Logic**: Automatically handles feature scaling and interaction terms.
