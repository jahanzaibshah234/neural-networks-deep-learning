# Neural Network Hyperparameter Tuning with TensorFlow

A systematic manual hyperparameter search across **21 controlled experiments** applied to bank customer churn prediction. Each experiment isolates one hyperparameter — learning rate, optimizer, architecture, batch size, dropout, or activation — while keeping all others fixed, enabling direct comparison of each factor's individual impact.

## Overview

Hyperparameter selection is one of the most critical steps in building effective neural networks. This project implements a rigorous **one-factor-at-a-time (OFAT)** search strategy on the Bank Customer Churn dataset (Churn_Modelling.csv). All experiments share the same preprocessing pipeline, SEED=42 for reproducibility, and EarlyStopping for fair epoch comparison.

**Best configuration found:**
```
Architecture: (64, 32, 16) | Optimizer: Adam | LR: 0.001
Activation: Tanh | Dropout: 0.2 | Batch Size: 32
→ Val Accuracy: 86.69% | Test Accuracy: 86.45%
```

## Experiment Design

### Baseline Configuration

| Hyperparameter | Baseline Value |
|---|---|
| Architecture | (64, 32, 16) |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Activation | ReLU |
| Dropout | 0.0 |
| Batch Size | 32 |
| Epochs | Up to 50 (EarlyStopping) |

### Search Space — 21 Total Experiments

| Hyperparameter | Values Tested | # Experiments |
|---|---|---|
| Learning Rate | 0.01, 0.001, 0.0001 | 3 |
| Optimizer | Adam, SGD, RMSProp | 3 |
| Architecture | (32,16,8), (64,32,16), (128,64,32), (128,64,32,16) | 4 |
| Batch Size | 16, 32, 64, 128 | 4 |
| Dropout Rate | 0.0, 0.2, 0.3, 0.5 | 4 |
| Activation | ReLU, Tanh | 2 |
| Baseline | Default config | 1 |
| **Total** | | **21** |

## Model Architecture

A `build_model()` factory function constructs a fresh Sequential model per experiment:

```
Input(11) → [Dense(units, activation) → Dropout(rate)?] × N_layers → Dense(1, Sigmoid)
```

- `tf.keras.backend.clear_session()` called before each build to prevent memory leaks
- Dropout inserted after each hidden layer **only when** `dropout_rate > 0`
- Optimizer instantiated dynamically: Adam / SGD / RMSProp
- Loss: Binary Crossentropy | Metric: Accuracy

### EarlyStopping (All Experiments)
```
monitor=val_loss | patience=8 | restore_best_weights=True
```

### Final Model (Best Config Retrain)
```
monitor=val_loss | patience=10 | epochs=100
```

## Dataset

**File:** `data/Churn_Modelling.csv` — 10,000 bank customer records

### Preprocessing Pipeline

1. Drop `RowNumber`, `CustomerId`, `Surname`
2. LabelEncode `Gender` → 0/1
3. One-hot encode `Geography` with `drop_first=True` → `Geography_Germany`, `Geography_Spain`
4. Stratified train/test split — 80/20 (`stratify=y`, `random_state=42`)
5. Stratified train/val split — 80/20 from training portion
6. `StandardScaler` fit on `X_train` only → transform `X_val`, `X_test`
7. Scaler saved as `models/scaker.pkl`

### Data Splits

| Split | Size |
|---|---|
| Train | ~6,400 samples |
| Validation | ~1,600 samples |
| Test | 2,000 samples |

## Results

### All Experiments Ranked by Validation Accuracy

| Experiment | Val Acc | Test Acc | F1 Score | Epochs |
|---|---|---|---|---|
| **Activation tanh ★** | **0.8669** | **0.8675** | **0.8539** | **43** |
| Architecture (32,16,8) | 0.8606 | 0.8635 | 0.8488 | 23 |
| Batch Size 128 | 0.8600 | 0.8610 | 0.8452 | 24 |
| Architecture (128,64,32) | 0.8594 | 0.8585 | 0.8462 | 13 |
| Baseline | 0.8581 | 0.8650 | 0.8511 | 16 |
| Dropout 0.0 | 0.8569 | 0.8645 | 0.8506 | 18 |
| Optimizer adam | 0.8569 | 0.8640 | 0.8507 | 17 |
| Batch Size 32 | 0.8556 | 0.8590 | 0.8444 | 14 |
| LR 0.01 | 0.8550 | 0.8620 | 0.8439 | 12 |
| Activation relu | 0.8550 | 0.8690 | 0.8567 | 29 |
| Dropout 0.5 | 0.8550 | 0.8635 | 0.8457 | 50 |
| Dropout 0.2 | 0.8550 | 0.8695 | 0.8563 | 30 |
| Batch Size 16 | 0.8550 | 0.8520 | 0.8413 | 12 |
| Dropout 0.3 | 0.8544 | 0.8605 | 0.8443 | 29 |
| LR 0.0001 | 0.8537 | 0.8665 | 0.8543 | 50 |
| LR 0.001 | 0.8531 | 0.8615 | 0.8493 | 15 |
| Batch Size 64 | 0.8531 | 0.8635 | 0.8508 | 25 |
| Architecture (128,64,32,16) | 0.8525 | 0.8625 | 0.8506 | 14 |
| Optimizer rmsprop | 0.8512 | 0.8625 | 0.8510 | 15 |
| Architecture (64,32,16) | 0.8512 | 0.8650 | 0.8517 | 15 |
| Optimizer sgd | 0.7962 | 0.7965 | 0.7063 | 50 |

### Best Configuration (Final Model)

| Parameter | Value |
|---|---|
| Architecture | (64, 32, 16) |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Activation | **Tanh** |
| Dropout | **0.2** |
| Batch Size | 32 |
| Test Accuracy | **86.45%** |
| Validation Accuracy | **86.69%** |
| Precision | 0.8605 |
| Recall | 0.8675 |
| F1 Score | 0.8539 |
| Test Loss | 0.3342 |
| Epochs | 43 |

### Baseline vs. Final Model

| Metric | Baseline | Final Tuned |
|---|---|---|
| Test Accuracy | 86.50% | 86.45% |
| Validation Accuracy | 85.81% | 86.69% |
| Activation | ReLU | Tanh |
| Dropout | 0.0 | 0.2 |
| Epochs | 16 | 43 |

## Key Findings

- **Tanh outperformed ReLU** — highest validation accuracy (86.69%) due to zero-centered outputs benefiting gradient flow on this feature distribution
- **SGD is unsuitable without tuning** — 79.65% vs Adam's 86.40% at the same lr=0.001; adaptive optimizers dominate
- **Dropout=0.2 consistently improves generalization** — test accuracy 86.95% vs 86.45% without dropout
- **Architecture width/depth shows diminishing returns** — all 4 architectures within ~0.65% of each other
- **Batch size trades speed for stability** — small batches converge faster but noisier; large batches smoother but need more epochs
- **All learning rates achieve >86%** — lower lr (0.0001) generalizes slightly better at cost of 50 epochs

## Visualizations

All plots are saved in the `result/` folder:

| File | Description |
|---|---|
| `result/hyperparameter_comparison.png` | Bar chart — top 10 experiments by validation accuracy |
| `result/accuracy_curve.png` | Final model train vs. validation accuracy |
| `result/loss_curve.png` | Final model train vs. validation loss |
| `result/confusion_matrix.png` | Final model confusion matrix on test set |

## Sample Customer Prediction

The saved model supports real-time inference:

```python
import joblib
import numpy as np
import tensorflow as tf

model  = tf.keras.models.load_model("models/best_churn_model.keras")
scaler = joblib.load("models/scaker.pkl")

sample = np.array([[650, 1, 40, 5, 100000, 2, 1, 1, 60000, 0, 0]])
sample_scaled = scaler.transform(sample)

prob = model.predict(sample_scaled)[0][0]
print(f"Churn Probability: {prob:.2%}")
print("CHURN" if prob >= 0.5 else "STAY")
```

## Project Structure

```
Neural_Network_Hyperparameter_Tuning_with_TensorFlow/
├── data/
│   └── Churn_Modelling.csv              # Bank customer churn dataset
├── models/
│   ├── best_churn_model.keras           # Saved final model
│   └── scaker.pkl                       # Saved StandardScaler
├── notebook/
│   └── Hyperparameter_Tuning.ipynb      # Main notebook (all 21 experiments)
├── report/
│   └── Hyperparameter_Tuning_Report.pdf # Full project report
├── result/
│   ├── experiment_results.csv           # All 21 experiment metrics
│   ├── best_model_configuration.csv     # Best config + final test accuracy
│   ├── hyperparameter_comparison.png    # Top 10 bar chart
│   ├── accuracy_curve.png               # Final model accuracy curve
│   ├── loss_curve.png                   # Final model loss curve
│   └── confusion_matrix.png             # Final model confusion matrix
├── requirements.txt                     # Python dependencies
└── README.md
```

## Tools and Technologies

- Python
- TensorFlow / Keras
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Joblib
- Jupyter Notebook

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Churn Modelling Dataset — Kaggle](https://www.kaggle.com/datasets/shubh0799/churn-modelling)
- Goodfellow, I. et al. (2016). *Deep Learning*. MIT Press.
