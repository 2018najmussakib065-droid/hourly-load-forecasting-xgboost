![XGBoost-and-its-Uses-in-Machine-Learning-1108x690](https://github.com/user-attachments/assets/756c157e-5819-4c36-8e36-fac8701070a4)

# Time Series Forecasting: Energy Consumption Analysis with XGBoost and Prophet
A comprehensive time series analysis project for forecasting hourly energy consumption in the PJM East region using advanced machine learning techniques, featuring extensive data preprocessing, feature engineering, and model evaluation.

## Project Overview
This project implements a comprehensive time series forecasting pipeline for predicting hourly energy consumption in megawatts (MW). It explores two powerful forecasting approaches:

**XGBoost Regression** - Gradient boosting with extensive feature engineering
**Facebook Prophet** - Automated time series forecasting with trend and seasonality decomposition

The analysis includes exploratory data analysis, feature engineering, model training, evaluation, and long-term forecasting visualization.
## Dataset
Source: PJM East Hourly Energy Consumption Data

Time Period: 2002-2020 (approximately 145,000+ hourly observations)

Target Variable: PJME_MW (Energy consumption in megawatts)

Frequency: Hourly measurements

## Data Characteristics:

Strong daily seasonality (business hours vs off-hours)

Weekly patterns (weekdays vs weekends)

Yearly seasonality (summer cooling and winter heating peaks)

Long-term growth trend

## Technologies Used

### Core Libraries:

Python 3.x
pandas - Data manipulation and analysis
NumPy - Numerical computing
scikit-learn - Machine learning utilities and metrics

### Machine Learning:

XGBoost - Gradient boosting framework

Facebook Prophet - Time series forecasting

### Visualization:

Matplotlib - Core plotting library

Seaborn - Statistical data visualization

### Data Processing:

pandas.api.types.CategoricalDtype - Ordered categorical features

## Key Features

### Data Preprocessing

✅ Datetime index parsing and validation

✅ Missing value detection and handling

✅ Data resampling to ensure continuous hourly intervals

✅ Train-test split with chronological ordering (crucial for time series)

✅ Outlier analysis and visualization

### Feature Engineering

✅ Temporal Features: hour, day of week, month, quarter, year, day of year, week of year

✅ Categorical Features: Weekday names (ordered), seasons (Spring/Summer/Fall/Winter)

✅ Lag Features: 1-hour, 24-hour, and 168-hour (1 week) lags

✅ Rolling Statistics: 24-hour moving averages and standard deviations

✅ Date Offset Calculation: Custom seasonal position encoding

## Model Implementation

### XGBoost Model

Conservative hyperparameter tuning (max_depth=3, learning_rate=0.01)
Early stopping to prevent overfitting (50 rounds)
Feature importance analysis using gain metric
Comprehensive evaluation metrics (RMSE, MAE, MAPE, R²)

### Prophet Model

Automatic seasonality detection (daily, weekly, yearly)
Trend decomposition with changepoint detection
Uncertainty interval estimation (95% confidence)
Component visualization (trend, seasonality, holidays)
Long-term forecasting (up to 5+ years)

## Visualizations

### 1. Raw Data Exploration

Full Time Series Plot: Dot plot (ms=1) showing ~18 years of hourly data

Weekly Zoom: Detailed view of one week to identify daily patterns

Train-Test Split: Visualization showing data partition

### 2. Feature Analysis

Box Plots: Energy consumption by weekday and season

> sns.boxplot(data=features_and_target, x='weekday', y='PJME_MW', 
              hue='season', linewidth=1)

Feature Importance: Horizontal bar chart ranking XGBoost features

Correlation Analysis: Heatmap of feature relationships

### 3. Model Performance

Predictions vs Actuals: Scatter plots and time series overlays

Residual Plots: Distribution and time series of prediction errors

Prophet Components: Decomposition showing trend, weekly, and yearly seasonality

Uncertainty Intervals: Shaded regions showing 95% confidence bounds

### 4. Forecast Visualization

Long-term Forecast: 5.5-year (2000-day) predictions with uncertainty

First Month Detail: Zoomed view showing hourly pattern accuracy

Component Breakdown: Trend, weekly, and yearly seasonality contributions

## Results and Insights

##Key Findings

### Strong Temporal Patterns:

Daily Cycle: Clear business hours (8 AM - 6 PM) peak consumption

Weekly Cycle: Weekdays consume 10-15% more than weekends

Seasonal Patterns: Summer and winter peaks due to HVAC usage


### Feature Importance (XGBoost):

Top 3 Features: lag24 (yesterday same time), dayofyear, hour

Lag features capture autoregressive nature (previous values predict future)

Time features capture cyclical patterns


### Model Comparison:

Prophet: Better overall performance (RMSE: 81.36 MW)

XGBoost: More flexible, allows custom features

Both models: Capture seasonality effectively

