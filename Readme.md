# Hourly Electricity Consumption Forecasting using XGBoost

Time Series Analysis · Machine Learning · Feature Engineering

This project applies machine learning-based time series forecasting to predict hourly electricity demand using the PJME (PJM East) hourly dataset.
Unlike traditional ARIMA/SARIMA models, this project uses XGBoost Regressor, enhanced by thoughtful feature engineering, train-test splitting by time, and model evaluation on unseen future data.

## Project Overview

The goal of this project is to forecast electricity load for operational and planning purposes.
Key steps include:

Data preprocessing & cleaning

Creating time-based features (hour, day, month, year, season, etc.)

Splitting data into chronological train/test sets

Training an XGBoost regression model

Evaluating performance on January 2015 load values

Visualizing predictions vs actuals

This provides a realistic demonstration of machine-learning forecasting for real-world time series.

## Features Used for Forecasting

The dataset was enhanced using custom feature engineering:

**Hour of day**

**Day of week**

**Month / Quarter / Year**

**Day of year**

**ISO week of year**

**Season (Spring, Summer, Fall, Winter)**

**Categorical encoding for weekdays**

These features help XGBoost learn both daily and seasonal patterns.

## Tech Stack

**Python**

**Pandas**

**NumPy**

**Matplotlib / Seaborn**

**XGBoost**

**Scikit-learn**

**Jupyter Notebook**

## Dataset
Dataset used: **PJME_hourly.csv**
It contains hourly electricity demand (MW) for the PJM East region.

## Methodology
### 1. Load & clean the dataset

Datetime index created:

> pjme = pd.read_csv("PJME_hourly.csv", parse_dates=['Datetime'], index_col='Datetime')

### 2. Feature Engineering

Custom function creates 10+ time-based features:

> X, y = create_features(pjme, label='PJME_MW')

### 3. Train–Test Split (Chronological)

Using **1-Jan-2015** as the cutoff date.

### 4. Model Training
> model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)

> model.fit(X_train, y_train)

### 5. Forecasting & Evaluation

Forecasts for **January 2015** were compared against actual values.

## Metrics used:

**MAE**

**RMSE**

**MAPE (Mean Absolute Percentage Error)**
>This helps measure real-world forecasting accuracy.

## Results (Summary)

XGBoost successfully modeled hourly trends.

Captured daily and weekly seasonality using feature engineering.

Forecast accuracy was reasonable, and visualization showed the model followed electricity load patterns closely.

## Visualization Example

Forecast vs Actuals for January 2015:

> ax.scatter(pjme_test.index, pjme_test['PJME_MW'], color='red')

> model.plot(pjme_test_fcst, ax=ax)

> plt.title("January 2015 Forecast vs Actuals")
