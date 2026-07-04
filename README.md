# Electric Vehicle Battery State of Health (SOH) Prediction Using Machine Learning

## Project Overview

This project predicts the State of Health (SOH) of an Electric Vehicle (EV) battery using Machine Learning algorithms. Battery SOH is an important indicator of battery performance and remaining life.

The project analyzes battery parameters, trains multiple machine learning models, compares their performance, and predicts the SOH for new battery data.

---

## Problem Statement

Develop a Machine Learning model to predict the State of Health (SOH) of an Electric Vehicle battery using battery operating parameters.

---

## Objectives

- Analyze battery dataset
- Perform data preprocessing
- Clean the dataset
- Train multiple Machine Learning models
- Compare model performance
- Select the best model
- Predict battery SOH

---

## Dataset Information

Dataset Size:
- 2000 Records
- 13 Columns

Features:

- BatteryID
- BatchID
- Cycle
- Voltage
- Current
- Temperature
- ChargeTime
- DischargeTime
- InternalResistance
- Capacity
- AmbientHumidity
- C_Rate

Target Variable:

- SOH (State of Health)

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## Machine Learning Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

---

##  Model Performance

| Model | MAE | RMSE | R² Score |
|---------|---------|---------|---------|
| Linear Regression | 0.8594 | 1.0805 | 0.9967 |
| Decision Tree Regressor | 1.1335 | 1.4356 | 0.9942 |
| Random Forest Regressor | 0.8219 | 1.0434 | 0.9969 |

Best Model:
**Random Forest Regressor**

## Project Structure

```text
EV_Battery_Health_Checker/
|
|-- dataset/
|-- models/
|-- src/
|   |-- data_preprocessing.py
|   |-- data_cleaning.py
|   |-- data_analysis.py
|   |-- train_model.py
|   |-- evaluate_model.py
|   |-- predict.py
|
|-- README.md
|-- requirements.txt
```

---

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python src/train_model.py
```

### Predict SOH

```bash
python src/predict.py
```

---

## Sample Prediction

```text
Predicted SOH: 62.49

Battery Condition:
Replace Soon
```

---

## Future Improvements

- Real-time battery monitoring
- Web application
- Mobile application
- IoT integration
- Deep Learning models

---

## Team Members

- Muthuiakshmi P
- Kaviyamaheshwari J
- Asimfathima P
- Yuktha S
