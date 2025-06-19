# 🏈 NFL Time Prediction Analysis

## Project Overview

This project implements a comprehensive machine learning pipeline to predict `MinutesElapsed_target` in NFL games using various ML models. The analysis combines traditional regression techniques with deep learning approaches to forecast game timing patterns.

**Author:** Fiorella Abarca  
**Purpose:** Predict NFL game timing using machine learning models

## 🎯 Key Features

- **Multiple ML Models**: Linear Regression, Ridge Regression, XGBoost, Neural Networks
- **Cross-Validation**: Robust model evaluation with 5-fold cross-validation
- **Feature Engineering**: NFL-specific data preprocessing and feature selection
- **Comprehensive Evaluation**: Multiple metrics (MAE, RMSE, R²) for thorough assessment
- **Model Persistence**: Save trained models for production use
- **Visualization**: Rich plots for data exploration and model diagnostics

## 📊 Dataset

- **Training Data**: `nfl_train.csv`
- **Validation Data**: `nfl_validation.csv`
- **Target Variable**: `MinutesElapsed_target`
- **Features**: Various NFL game statistics including downs, yard lines, scores, and game timing

## 🛠️ Technologies Used

```python
# Core Libraries
pandas, numpy, matplotlib, seaborn

# Machine Learning
scikit-learn, xgboost

# Deep Learning
tensorflow/keras

# Model Persistence
joblib
```

## 📁 Project Structure

```
nfl-prediction/
├── nfl_predictionv2.py          # Main analysis script
├── nfl_train.csv                # Training dataset
├── nfl_validation.csv           # Validation dataset
├── outputs/
│   ├── nfl_benchmark_results.csv
│   ├── nfl_validation_predictions.csv
│   ├── nfl_scaler.pkl
│   ├── nfl_*.pkl                # Trained ML models
│   └── nfl_neural_network_model.h5
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
```

### Running the Analysis

```python
# Execute the complete pipeline
python nfl_predictionv2.py

# Or run in Jupyter Notebook
jupyter notebook nfl_predictionv2.ipynb
```

## 📈 Model Performance

The project evaluates multiple models using cross-validation:

| Model | CV MAE | Test MAE | Test RMSE | Test R² |
|-------|--------|----------|-----------|---------|
| Linear Regression | ~X.XX | X.XX | X.XX | X.XX |
| Ridge Regression | ~X.XX | X.XX | X.XX | X.XX |
| XGBoost | ~X.XX | X.XX | X.XX | X.XX |
| Neural Network | ~X.XX | X.XX | X.XX | X.XX |

*Note: Actual values depend on your dataset*

## 🔍 Key Analysis Steps

### 1. Data Preprocessing
- Remove irrelevant columns (`ScoreType.1`, `DataSet`, `custom_index`)
- Drop low-correlation features (`Half`, `YardsToFirstDown`, `MarginOffense`)
- Outlier detection and analysis
- Feature scaling with MinMaxScaler

### 2. Exploratory Data Analysis
- Distribution analysis of target variable
- Correlation heatmaps
- Outlier visualization
- Data quality assessment

### 3. Model Training
- **Linear Models**: Baseline and regularized approaches
- **Ensemble**: XGBoost for handling non-linear patterns
- **Deep Learning**: Multi-layer neural network with early stopping
- **Cross-Validation**: 5-fold CV for reliable performance estimates

### 4. Model Evaluation
- Multiple metrics for comprehensive assessment
- Residual analysis for model diagnostics
- Validation on unseen data
- Model comparison and selection

### 5. Production Pipeline
- Model serialization for deployment
- Prediction generation on validation set
- Performance benchmarking across datasets

## 📊 Output Files

The analysis generates several output files:

- **`nfl_benchmark_results.csv`**: Complete model comparison results
- **`nfl_validation_predictions.csv`**: Predictions on validation data
- **Model Files**: Serialized models for production use
- **`nfl_scaler.pkl`**: Fitted data scaler for consistent preprocessing

## 🎯 Key Insights

### Target Variable Analysis
- Mean: ~XX minutes
- Contains negative values (~XX%) indicating clock adjustments
- Non-normal distribution requiring robust modeling approaches

### Feature Importance
- Most predictive features: [Based on your correlation analysis]
- Low-impact features removed during preprocessing
- NFL-specific patterns captured through domain knowledge

## 🔧 Model Configuration

### Neural Network Architecture
```python
# 2 Hidden Layers + Output Layer
Layer 1: Dense(n_features, activation='relu')
Layer 2: Dense(n_features, activation='relu') 
Output:  Dense(1, activation=None)

# Training Configuration
Optimizer: Adam
Loss: MSE
Early Stopping: 30 patience epochs
Batch Size: 128
```

### XGBoost Configuration
```python
XGBRegressor(
    random_state=42,
    n_estimators=100
    # Additional hyperparameters can be tuned
)
```

## 📝 Usage Examples

### Loading Trained Models
```python
import joblib
from tensorflow.keras.models import load_model

# Load ML models
ridge_model = joblib.load('nfl_ridge_model.pkl')
xgb_model = joblib.load('nfl_xgbregressor_model.pkl')

# Load scaler
scaler = joblib.load('nfl_scaler.pkl')

# Load neural network
nn_model = load_model('nfl_neural_network_model.h5')
```

### Making Predictions
```python
# Prepare new data
X_new_scaled = scaler.transform(X_new)

# Get predictions
ridge_pred = ridge_model.predict(X_new_scaled)
xgb_pred = xgb_model.predict(X_new_scaled)
nn_pred = nn_model.predict(X_new_scaled)
```

## 📋 Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
tensorflow>=2.6.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request


## 📞 Contact

**Fiorella Abarca**
- Email: fabarcavalñverde@gmail.com
- LinkedIn: (https://www.linkedin.com/in/fabarcavalverde/)
- GitHub: (https://github.com/Fabarcavalverde)


---

*Last Updated: [16/6/2025]
*Version: 2.0*