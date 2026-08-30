# 🌡️ Hungary Temperature Trend & Time-Series Forecasting

Egy end-to-end Python adattudományi és gépi tanulási projekt, amely Magyarország historikus havi átlaghőmérsékleti adatait elemzi, vizualizálja a szezonalitást és az éghajlati anomáliákat, valamint gépi tanulási regressziós modellekkel jósolja meg a következő hónap várható hőmérsékletét.

---

## 📌 A Projekt Összefoglalása

* **Domain:** Idősoros elemzés és predikció (Time-Series Forecasting)
* **Adatforrás:** Magyarország havi átlaghőmérsékleti adatai (1901–2013)
* **Cél:** Szezonális trendek feltárása, lag-jellemzők (Lag Features) építése és 1-lépéses előrejelzés (One-step-ahead forecasting) Scikit-Learn modellek segítségével.

---

## 📊 Fő Eredmények & Modell Teljesítmény

A modellek értékelése szigorú idősoros vágású (Time-Series Split, `shuffle=False`) tesztkészleten történt, elkerülve az adatszivárgást. A nem-lineáris összefüggéseket és a szezonalitást a Random Forest Regressor kezelte a legsikeresebben:

| Modell | MAE (°C) | RMSE (°C) | R² Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 1.675 | 2.249 | 0.919 |
| **Random Forest Regressor** | **1.515** | **2.068** | **0.931** |

> 💡 **Fő megállapítás:** A lag-jellemzők (`temp_lag_1` és `temp_lag_12`) bevonása drasztikusan növelte a modell pontosságát, lehetővé téve, hogy a Random Forest a variancia több mint 93%-át megmagyarázza.

---

## 🛠️ Alkalmazott Technológiák

* **Nyelv:** Python 3.x
* **Adatkezelés & EDA:** Pandas, NumPy
* **Vizualizáció:** Seaborn, Matplotlib (FacetGrid, Anomália diagramok, Korrelációs mátrixok)
* **Machine Learning:** Scikit-Learn (LinearRegression, RandomForestRegressor, Train-Test Split, Metrics)

---

## 📂 Projekt Struktúra

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
   - Idősoros rendezés & Adattisztítás: Szigorú kronológiai sorrend biztosítása dátum alapján.
   - Lag Features:
       - temp_lag_1: Az előző hónap tényleges hőmérséklete ($t-1$).
         
       - temp_lag_12: Az előző év azonos hónapjának hőmérséklete ($t-12$).Rolling Window:temp_roll_mean_3: Az elmúlt 3 hónap gördülő átlaga (kizárva a tárgyhót az adatszivárgás elkerülésére).
         
       - Szezonális kódolás: One-Hot Encoding az évszakokra (season_Spring, season_Summer, season_Winter).

## ⚠️  Modell Architektúra & Korlátok (Model Limitations)Előrejelzési horizont (One-step-ahead Forecasting): 
  - A modell felépítéséből adódóan 1 hónapos előrejelzésre ($t+1$) lett optimalizálva. Mivel a predikció erősen támaszkodik a lag-jellemzőkre, a modell csak akkor ad pontos becslést, ha rendelkezésre állnak a közvetlenül megelőző hónapok valós mérései.
  
  - Adatállomány időbeli terjedelme: A historikus adatsor 2013 szeptemberéig tartalmaz adatokat. Ennek következtében a modell jelenlegi formájában a 2013 utáni távoli időszakokra (pl. 2026/2027-re) nem alkalmazható közvetlenül valós input adatok hiányában.
  
  - Hosszú távú (Multi-step) jóslás korlátai: A rekurzív (lépésről lépésre történő) előrejelzés több évre előre nem javasolt ezzel az architektúrával, mert a bejósolt értékekből számolt lag-változók miatt a hiba eltolódik és felhalmozódik (Autoregressive Error Accumulation).

## Dataset Info

The dataset is not included in the repository due to its size.

### Download

Download the original dataset from: https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data

### Setup

Place the downloaded file in:

data/raw/dataset.csv

## 🚀 Futtatás és Telepítés

```code
git clone [https://github.com/Hegeda15/hungary-temperature-ml.git](https://github.com/FELHASZNALONEV/hungary-temperature-ml.git)
cd hungary-temperature-ml
python -m venv venv
source venv/bin/activate  # Windows esetén: venv\Scripts\activate
pip install -r requirements.txt
