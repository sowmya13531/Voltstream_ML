# ⚡ VoltStream – ML & Predictive Analytics Platform

> Machine Learning-powered Energy Intelligence System built using Python, Scikit-Learn, XGBoost, Prophet, ARIMA, and the UCI Household Power Consumption Dataset.

---

## 📖 Overview

VoltStream Week 2 extends the AWS Serverless Energy Intelligence Platform developed in Week 1 by integrating a complete Machine Learning and Predictive Analytics layer.

The project implements **23 Machine Learning models across 5 AI paradigms** to analyze household energy consumption, predict future usage, detect anomalies, classify high-consumption days, identify consumption patterns, and forecast future electricity bills. 

The system is designed for **Prosumers** (users who both consume and generate electricity) and provides intelligent insights that can be integrated into smart energy management platforms.

---

# 🚀 Key Features

## 📈 Energy Consumption Prediction

Predict daily household energy consumption using:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Polynomial Regression
* Decision Tree Regression
* Random Forest Regression
* Gradient Boosting Regression
* XGBoost Regression
* Support Vector Regression (SVR)
* KNN Regression

Total Regression Models: **10** 

---

## 🚨 High Usage Day Classification

Predict whether a day will exceed normal energy consumption levels.

Models include:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier
* XGBoost Classifier
* Support Vector Machine
* KNN Classifier
* Gaussian Naive Bayes

Total Classification Models: **8** 

---

## 🔍 Behavioral Pattern Discovery

Identify household energy usage groups using:

* K-Means Clustering

Discover patterns such as:

* Low Consumption Days
* Moderate Consumption Days
* Peak Consumption Days



---

## ⚠️ Anomaly Detection

Detect unusual energy consumption events using:

* DBSCAN

Examples:

* Appliance malfunction
* Unexpected occupancy
* Energy spikes
* Abnormal billing events



---

## 🧠 Neural Networks

Deep learning-based predictions using:

* MLP Regressor
* MLP Classifier

Multi-layer neural network architectures with Adam optimization and early stopping. 

---

## 📅 Time Series Forecasting

Forecast future electricity costs and usage using:

### ARIMA

* Monthly electricity bill forecasting
* 6-month prediction horizon

### Prophet

* Daily bill forecasting
* 120-day future prediction
* Trend analysis
* Seasonal decomposition



---

# 🏗️ Machine Learning Architecture

```text
                    Household Power Dataset
                               │
                               ▼

                ┌────────────────────────────┐
                │ Data Cleaning & Processing │
                └────────────────────────────┘
                               │
                               ▼

                ┌────────────────────────────┐
                │ Feature Engineering & EDA  │
                └────────────────────────────┘
                               │
                               ▼

                  Daily Aggregated Dataset
                          (1442 Rows)
                               │
                               ▼

      ┌─────────────────────────────────────────────┐
      │            Machine Learning Layer           │
      └─────────────────────────────────────────────┘

             │             │            │
             ▼             ▼            ▼

      Regression     Classification  Clustering
      (10 Models)      (8 Models)    (1 Model)

             │             │            │
             └─────────────┼────────────┘
                           ▼

                 Anomaly Detection
                     (DBSCAN)

                           ▼

                    Neural Networks
                    (MLP Models)

                           ▼

                  Time Series Models
                ARIMA + Prophet Forecast

                           ▼

                    Energy Insights
```

Based on the Week 2 ML architecture pipeline. 

---

# 📊 Dataset

This project uses the UCI Household Power Consumption Dataset.

### Dataset Details

| Property  | Value                           |
| --------- | ------------------------------- |
| Dataset   | UCI Household Power Consumption |
| Duration  | Dec 2006 – Nov 2010             |
| Records   | ~2,075,259                      |
| Columns   | 9                               |
| Frequency | Minute-Level                    |
| Format    | Semicolon-separated TXT         |



---

## Dataset Features

| Feature               | Description                       |
| --------------------- | --------------------------------- |
| Global_active_power   | Total household power consumption |
| Global_reactive_power | Reactive energy                   |
| Voltage               | Line voltage                      |
| Global_intensity      | Current draw                      |
| Sub_metering_1        | Kitchen appliances                |
| Sub_metering_2        | Laundry appliances                |
| Sub_metering_3        | HVAC and water heater             |



---

# ⚙️ Feature Engineering

The raw dataset is transformed into daily energy intelligence features.

### Engineered Features

* DayOfWeek
* Month
* Quarter
* IsWeekend
* Daily_kWh
* Bill_INR
* HighUsage

The HighUsage feature is used as the classification target, while Bill_INR is used for energy cost forecasting. 

---

# 📈 Exploratory Data Analysis (EDA)

Analysis performed:

* Energy consumption trends
* Seasonal usage patterns
* Monthly comparisons
* Correlation heatmaps
* Distribution analysis
* Sub-metering comparisons

Key findings:

* Winter months show higher consumption
* HVAC systems consume the most energy
* Global_intensity strongly correlates with active power
* Voltage decreases during heavy loads



---

# 🤖 Machine Learning Models

## Regression Models

| Model                       |
| --------------------------- |
| Linear Regression           |
| Ridge Regression            |
| Lasso Regression            |
| Polynomial Regression       |
| Decision Tree Regressor     |
| Random Forest Regressor     |
| Gradient Boosting Regressor |
| XGBoost Regressor           |
| SVR (RBF Kernel)            |
| KNN Regressor               |

Target:

```text
Global_active_power
```

Evaluation Metrics:

* RMSE
* MAE
* R² Score



---

## Classification Models

| Model                |
| -------------------- |
| Logistic Regression  |
| Decision Tree        |
| Random Forest        |
| Gradient Boosting    |
| XGBoost              |
| SVM                  |
| KNN                  |
| Gaussian Naive Bayes |

Target:

```text
HighUsage
```

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC



---

## Unsupervised Learning

### K-Means Clustering

Purpose:

* Customer behavior segmentation
* Energy usage profiling
* Personalized recommendations

Configuration:

```text
Clusters = 3
```



---

## Anomaly Detection

### DBSCAN

Purpose:

* Detect unusual energy usage
* Identify appliance faults
* Billing anomaly detection

Parameters:

```python
eps = 0.5
min_samples = 5
```



---

## Neural Networks

### MLP Regressor

Architecture:

```text
256 → 128 → 64 → 32
```

### MLP Classifier

Architecture:

```text
128 → 64 → 32
```

Features:

* Adam Optimizer
* Early Stopping
* ReLU Activation



---

# ⏳ Time Series Forecasting

## ARIMA Forecasting

Configuration:

```text
ARIMA(2,1,2)
```

Purpose:

* Monthly electricity bill forecasting
* 6-month future prediction



---

## Prophet Forecasting

Configuration:

```text
seasonality_mode = multiplicative
yearly_seasonality = True
weekly_seasonality = True
```

Purpose:

* Daily bill prediction
* 120-day forecasting horizon
* Trend and seasonality analysis



---

# 📂 Project Structure

```text
Voltstream_ML/
│
├── Voltstream_ML.ipynb
├── household_power_consumption.txt
├── README.md
│
├── data/
│   └── household_power_consumption.txt
│
├── outputs/
│   ├── regression_results/
│   ├── classification_results/
│   ├── clustering/
│   ├── anomaly_detection/
│   └── forecasts/
│
└── requirements.txt
```

---

# 🛠️ Technology Stack

## Machine Learning

* Scikit-Learn
* XGBoost
* Prophet
* Statsmodels

## Data Processing

* Pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn

## Development Environment

* Google Colab
* Python 3.x



---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/sowmya13531/Voltstream_ML.git

cd Voltstream_ML
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

Or

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost prophet statsmodels
```



---

# ▶️ Running the Project

## Open Notebook

```bash
jupyter notebook Voltstream_ML.ipynb
```

or upload to Google Colab.

---

## Dataset Upload

Upload:

```text
household_power_consumption.txt
```

Then execute notebook cells sequentially.



---

# 📊 Evaluation Metrics

## Regression

* RMSE
* MAE
* R² Score

## Classification

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

## Forecasting

* RMSE

 

---

# 🎯 Business Use Cases

### Energy Consumption Prediction

Predict tomorrow's household energy demand.

### Budget Alerts

Identify high-consumption days before they occur.

### Smart Recommendations

Group users into consumption profiles.

### Fault Detection

Detect unusual appliance behavior.

### Electricity Bill Forecasting

Forecast monthly and daily electricity expenses.



---

# 📈 ML Coverage Summary

| Category                | Models |
| ----------------------- | ------ |
| Regression              | 10     |
| Classification          | 8      |
| Clustering              | 1      |
| Anomaly Detection       | 1      |
| Neural Networks         | 2      |
| Time Series Forecasting | 2      |
| Total Models            | 23     |



---

# 🎬 Demo Flow

1. Load UCI dataset
2. Show data cleaning process
3. Demonstrate feature engineering
4. Display EDA visualizations
5. Run XGBoost Regressor
6. Run XGBoost Classifier
7. Show K-Means clusters
8. Demonstrate DBSCAN anomalies
9. Show Prophet forecast
10. Display final model leaderboard



---


Tachyon AIML Internship 2026 – ML & Predictive Analytics Track


