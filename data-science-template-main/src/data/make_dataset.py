import os
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
   print(f"Loading data from {file_path}")
   df = pd.read_csv(file_path)
   return df

def clean_temperature_data(df: pd.DataFrame) -> pd.DataFrame:
   print("Cleaning in progress...")

   df_clean= df.copy()

   df_clean['dt'] = pd.to_datetime(df_clean['dt'])
   df_clean = df_clean.sort_values('dt').reset_index(drop=True)

   df_clean = df_clean.dropna(subset=['AverageTemperature'])

   df_clean['year'] = df_clean['dt'].dt.year
   df_clean['month'] = df_clean['dt'].dt.month
   df_clean['day'] = df_clean['dt'].dt.day

   
   return df_clean


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
   print(f"Saving cleaned data to {output_path}")
   df.to_csv(output_path, index=False)

   print("Data cleaning and saving completed successfully.")
   

def main():
   raw_data_path ="data", "raw", "GlobalLandTemperaturesByCity.csv"
   interim_data_path = 'data/interim/hungary_temperatures_clean.csv'

   df_raw = load_data(os.path.join(*raw_data_path))

   df_hun= df_raw[df_raw['Country'] == 'Hungary'].copy()
   
   df_cleaned = clean_temperature_data(df_hun)

   save_cleaned_data(df_cleaned, interim_data_path)


if __name__ == "__main__":
   main()
