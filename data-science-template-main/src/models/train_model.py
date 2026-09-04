import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def load_data(file_path: str) -> pd.DataFrame:
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Preparing data...")
    df_ml = pd.get_dummies(df, columns=['season'], dtype=int)
  
    feature_cols = [
        'year',
        'month',
        'temp_lag_1',
        'temp_lag_12',
        'temp_roll_mean_3',
        'season_Spring',
        'season_Summer',
        'season_Winter',
    ]
    for col in feature_cols:
        if col not in df_ml.columns:
            df_ml[col] = 0

    X = df_ml[feature_cols]
    y = df_ml['AverageTemperature']
    return X, y

def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    """Idősoros vágással felosztja az adatot, betanítja a modellt és kiértékeli."""
    # Idősoros vágás (shuffle=False elengedhetetlen az idősoroknál!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print('Modell tanítása (RandomForestRegressor)...')
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predikció és metrikák
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print('\n====================================')
    print('  Random Forest Teljesítmény (Test) ')
    print('====================================')
    print(f'  MAE:  {mae:.3f} °C')
    print(f'  RMSE: {rmse:.3f} °C')
    print(f'  R²:   {r2:.3f}')
    print('====================================\n')

    return rf_model


def save_model(model, filepath: str) -> None:
    """Elmenti a betanított modellt joblib formátumban."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f'Modell sikeresen elmentve ide: {filepath}')


def main():
    processed_path = 'data/processed/hungary_temperatures_processed.csv'
    model_output_path = 'models/random_forest_model.pkl'

    df = load_data(processed_path)
    X, y = prep_data(df)
    model = train_and_evaluate(X, y)
    save_model(model, model_output_path)


if __name__ == '__main__':
    main()
