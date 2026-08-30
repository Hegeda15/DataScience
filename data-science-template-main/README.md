# 🌡️ Hungary Temperature Trend & Time-Series Forecasting

An end-to-end Python data science and machine learning project that analyzes historical monthly average temperature data for Hungary, visualizes seasonality and climate anomalies, and uses machine learning regression models to predict the expected temperature for the coming month.

---

## 📌 Project Summary

* **Domain:** Time-Series Forecasting
* **Adatforrás:** Monthly Average Temperature Data for Hungary (1901–2013)
* **Goal:** Identifying seasonal trends, building lag features, and performing one-step-ahead forecasting using Scikit-Learn models.
---

## 📊 Key Results & Model Performance

The models were evaluated on a test set with a strict time-series split (`shuffle=False`) to avoid data leakage. The Random Forest Regressor handled nonlinear relationships and seasonality most successfully:

| Modell | MAE (°C) | RMSE (°C) | R² Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 1.675 | 2.249 | 0.919 |
| **Random Forest Regressor** | **1.515** | **2.068** | **0.931** |

> 💡 **Key finding:** Including the lag features (`temp_lag_1` and `temp_lag_12`) dramatically improved the model's accuracy, enabling the Random Forest to explain more than 93% of the variance.

---

## 🛠️ Technologies Used

* **Nyelv:** Python 3.x
* **Adatkezelés & EDA:** Pandas, NumPy
* **Vizualizáció:** Seaborn, Matplotlib (FacetGrid, Anomália diagramok, Korrelációs mátrixok)
* **Machine Learning:** Scikit-Learn (LinearRegression, RandomForestRegressor, Train-Test Split, Metrics)

---

## 📂 Project Structure

```text
├── data/
│   ├── raw/                  # Eredeti, nyers adatállományok
│   └── processed/            # Tisztított és előkészített adatok
├── notebooks/
│   ├── 01_data_cleaning.ipynb      # Adattisztítás, típus-átalakítások
│   ├── 02_feature_engineering.ipynb# Lag features, rolling mean, évszakkódolás
│   ├── 03_eda_visualization.ipynb   # Trendek, anomáliák és szezonalitás ábrázolása
│   └── 04_modeling.ipynb           # Model tanítás, kiértékelés és metrikák
├── README.md                 # Projekt dokumentáció
└── requirements.txt          # Szükséges Python csomagok
```

## ⚙️  Feature Engineering & Módszertan 
   - Time Series Sorting & Data Cleaning: Ensuring strict chronological order based on date.
   - Lag Features:
       - temp_lag_1: The actual temperature of the previous month ($t-1$).

       - temp_lag_12: The temperature of the same month in the previous year ($t-12$). Rolling Window: temp_roll_mean_3: The rolling average of the past 3 months (excluding the current month to avoid data leakage).

       - Seasonal Encoding: One-Hot Encoding for seasons (season_Spring, season_Summer, season_Winter).

## ⚠️  Model Architecture & Limitations (Model Limitations) Forecasting Horizon (One-step-ahead Forecasting):
  - Due to the model’s structure, it has been optimized for a 1-month forecast ($t+1$). Since the prediction relies heavily on lag characteristics, the model provides accurate estimates only when actual measurements from the immediately preceding months are available.

  - Temporal scope of the dataset: The historical data series contains data through September 2013. Consequently, in its current form, the model cannot be directly applied to periods far beyond 2013 (e.g., 2026/2027) due to the lack of actual input data.

  - Limitations of long-term (multi-step) forecasting: Recursive (step-by-step) forecasting several years into the future is not recommended with this architecture because errors shift and accumulate (Autoregressive Error Accumulation) due to lag variables calculated from the forecasted values.

## Dataset Info

The dataset is not included in the repository due to its size.

### Download

Download the original dataset from: https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data

### Setup

Place the downloaded file in:

data/raw/dataset.csv

## 🚀 Run & installation

```code
git clone [https://github.com/Hegeda15/hungary-temperature-ml.git](https://github.com/FELHASZNALONEV/hungary-temperature-ml.git)
cd hungary-temperature-ml
python -m venv venv
source venv/bin/activate  # Windows esetén: venv\Scripts\activate
pip install -r requirements.txt
