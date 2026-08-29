#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[69]:


df = pd.read_csv("../data/processed/hungary_temperatures_processed.csv")
df.head()


# In[107]:


df.tail()


# In[70]:


df_ml= pd.get_dummies(df,columns=['season'],drop_first=True,dtype=int)


# In[71]:


predicted_cols=[
    'year',
    'month',
    'temp_lag_1',
    'temp_lag_12',
    'temp_roll_mean_3',
    'season_Spring',
    'season_Summer',
    'season_Winter',
]

X=df_ml[predicted_cols]
y=df_ml['AverageTemperature']


# In[89]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)


# In[99]:


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)


# In[100]:


y_pred_lr=lr_model.predict(X_test)
y_pred_rf=rf_model.predict(X_test)


def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f'=== {model_name} Eredmények ===')
    print(f'MAE:  {mae:.3f} °C')
    print(f'RMSE: {rmse:.3f} °C')
    print(f'R²:   {r2:.3f}\n')


evaluate_model(y_test, y_pred_lr, 'Linear Regression')
evaluate_model(y_test, y_pred_rf, 'Random Forest')


# In[101]:


plt.figure(figsize=(14, 5), dpi=150)

# Csak az utolsó 50 hónapot ábrázoljuk a tisztább láthatóságért
plt.plot(
    y_test.values[-50:],
    label='Tényleges (Actual)',
    color='black',
    linewidth=2,
)
plt.plot(
    y_pred_rf[-50:],
    label='Random Forest Jóslat',
    color='red',
    linestyle='--',
    linewidth=1.5,
)

plt.title(
    'Tényleges vs. Bejósolt Hőmérséklet (Teszt készlet)',
    fontsize=12,
    fontweight='bold',
)
plt.xlabel('Hónapok (Idősoros teszt szakasz)')
plt.ylabel('Hőmérséklet (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# # Test a 2027 év átlag hömérséklete

# In[109]:


# 1. Lekérjük az utolsó ismert 12 hónapot a datasetből (2012. szep - 2013. aug)
last_12_months = df.tail(12).reset_index(drop=True)

# 2. Bemeneti jellemzők összeállítása 2013 SZEPTEMBERÉRE
# - temp_lag_1: 2013. augusztus tényadat
# - temp_lag_12: 2012. szeptember tényadat
# - temp_roll_mean_3: 2013 június, július, augusztus átlaga
september_2013_input = pd.DataFrame(
    [
        {
            'year': 2013,
            'month': 9,
            'temp_lag_1': last_12_months.iloc[-1]['AverageTemperature'],
            'temp_lag_12': last_12_months.iloc[0]['AverageTemperature'],
            'temp_roll_mean_3': last_12_months['AverageTemperature']
            .iloc[-3:]
            .mean(),
            'season_Spring': 0,
            'season_Summer': 0,
            'season_Winter': 0,
        }
    ]
)

# Oszlopok beállítása az X_train sorrendjében
september_2013_input = september_2013_input[X_train.columns]

# 3. Jóslás lefuttatása
pred_sept_2013 = rf_model.predict(september_2013_input)[0]

print(f'=== 1-LÉPÉSES TESZT PREDIKCIÓ (2013. SZEPTEMBER) ===')
print(f'Bejósolt érték: {pred_sept_2013:.2f} °C')


# In[ ]:




