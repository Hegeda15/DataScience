import os
import pandas as pd

def load_interim_data(file_path: str) -> pd.DataFrame:
    print(f"Loading interim data from {file_path}")
    df = pd.read_csv(file_path)
    df['dt'] = pd.to_datetime(df['dt'])
    df = df[df['year'] >= 1900].copy()
    return df.sort_values('dt').reset_index(drop=True)

def ceate_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Creating lag features for the dataset")
    df_lagged = df.copy()

    df_lagged['temp_lag_1'] = df_lagged['AverageTemperature'].shift(1)

    df_lagged['temp_lag_12'] = df_lagged['AverageTemperature'].shift(12)

    df_lagged['temp_roll_mean_3'] = (
        df_lagged['AverageTemperature'].shift(1).rolling(window=3).mean()
    )

    return df_lagged

def create_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """Létrehozza az évszak kódolásokat."""
    df_season = df.copy()

    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'

    if 'season' not in df_season.columns:
        df_season['season'] = df_season['month'].apply(get_season)

    mean_temp = df_season['AverageTemperature'].mean()
    df_season['is_above_mean'] = (
        df_season['AverageTemperature'] > mean_temp
    ).astype(int)

    return df_season

def save_features(df: pd.DataFrame, output_path: str) -> None:

    print(f"Saving features data to {output_path}")

    df_ml = df.dropna().reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_ml.to_csv(output_path, index=False)
    print("Feature data saving completed successfully.")

def main():
   interim_data_path = 'data/interim/hungary_temperatures_clean.csv'
   features_data_path = 'data/processed/hungary_temperatures_processed.csv'

   df_interim = load_interim_data(interim_data_path)
   df_lagged = ceate_lag_features(df_interim)
   df_features = create_season_features(df_lagged)
   save_features(df_features, features_data_path)

if __name__ == "__main__":
   main()