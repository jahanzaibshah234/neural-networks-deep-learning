# Heart Disease Prediction Using Neural Networks and TensorFlow

An Artificial Neural Network (ANN) built with TensorFlow/Keras that predicts the presence of heart disease from patient health attributes.

## Overview

Heart disease is one of the leading causes of death worldwide, and early detection can significantly improve patient outcomes. This project implements a binary classification model that analyzes patient medical attributes (age, blood pressure, cholesterol, ECG results, etc.) and predicts whether heart disease is present.

## Model Architecture

```
Input Layer → Dense(64, ReLU) → Dropout(0.1) → Dense(32, ReLU) → Dense(1, Sigmoid)
```

| Layer | Units / Rate | Activation |
|---|---|---|
| Hidden Layer 1 (Dense) | 64 Neurons | ReLU |
| Dropout | Rate = 0.1 | — |
| Hidden Layer 2 (Dense) | 32 Neurons | ReLU |
| Output Layer (Dense) | 1 Neuron | Sigmoid |

- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Epochs:** 50
- **Batch Size:** 16
- **Validation Split:** 20%

## Dataset

The dataset contains patient medical records with the following features:

| Feature | Description |
|---|---|
| Age | Age of the patient |
| Sex | Gender of the patient |
| Chest Pain Type | Type of chest pain experienced |
| Resting Blood Pressure | Blood pressure while resting |
| Cholesterol | Serum cholesterol level |
| Fasting Blood Sugar | Blood sugar level |
| Resting ECG | Electrocardiographic results |
| Maximum Heart Rate | Maximum heart rate achieved |
| Exercise Induced Angina | Exercise-related chest pain |
| ST Depression | Depression induced by exercise |
| Slope | Slope of peak exercise ST segment |
| Number of Vessels | Major vessels colored by fluoroscopy |
| Thalassemia | Blood disorder information |
| **Target** | 0 = No Heart Disease, 1 = Heart Disease Present |

Data is preprocessed with `StandardScaler` and split 80/20 into training and testing sets.

## Results

**Test Accuracy: 95.12%**

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (No Disease) | 0.93 | 0.97 | 0.95 | 102 |
| 1 (Disease Present) | 0.97 | 0.93 | 0.95 | 103 |
| **Accuracy** | | | **0.95** | 205 |
| **Macro Avg** | 0.95 | 0.95 | 0.95 | 205 |
| **Weighted Avg** | 0.95 | 0.95 | 0.95 | 205 |

### Confusion Matrix

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 99 | 3 |
| **Actual 1** | 7 | 96 |

![Confusion Matrix](confusion_matrix.png)

### Training vs. Validation Accuracy

Training accuracy converges to ~0.98–0.99 while validation accuracy stabilizes around 0.93–0.95 over 50 epochs, indicating good convergence with limited overfitting thanks to the Dropout layer.

![Accuracy Curve](accuracy_curve.png)

## Tools and Technologies

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Jupyter Notebook

## Project Structure

```
Heart_Disease_Prediction/
├── data/
│   └── heart.csv
├── models/
│   └── heart_model.keras
├── notebook/
│   └── heart_disease.ipynb
├── reports/
│   └── Heart_Disease_Prediction_Report.pdf
└── requirements.txt
```

## Future Improvements

- Increase dataset size
- Further hyperparameter tuning
- Experiment with additional Dropout layers or rates
- Use deeper architectures
- Apply cross-validation

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Keras Documentation](https://keras.io/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
