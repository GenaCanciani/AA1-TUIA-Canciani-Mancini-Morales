import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
from sklearn.metrics import classification_report

# Normalizamos los nombres de las columnas 
COLS_FINALES = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm',
    'RainToday', 'season_spring', 'season_summer', 'season_winter',
    'region_East_Coast', 'region_Monsoonal_North', 'region_Murray_Basin',
    'region_Rangelands', 'region_Southern_Slopes',
    'region_Southern_and_South_Western_Flatlands', 'region_Wet_Tropics',
    'Month_sin', 'Month_cos', 'TempRange', 'HumidityDiff', 'PressureDiff',
    'WindDir9am_sin', 'WindDir9am_cos', 'WindDir3pm_sin', 'WindDir3pm_cos',
    'WindGustDir_sin', 'WindGustDir_cos'
]

def preprocesar_datos(df_raw):
    df = df_raw.copy()

    # Cargar y mergear regiones
    if os.path.exists('ciudades_regiones.csv'):
        regiones = pd.read_csv('ciudades_regiones.csv')
        if 'label' in regiones.columns:
            regiones = regiones.rename(columns={'label': 'region'})
        
        ciudades_corregidas = {
             'BadgerysCreek': 'Badgerys Creek', 'CoffsHarbour': 'Coffs Harbour',
             'NorahHead': 'Norah Head', 'NorfolkIsland': 'Norfolk Island',
             'WaggaWagga': 'Wagga Wagga', 'MountGinini': 'Mount Ginini',
             'Brisbane': 'Brisbane', 'MountGambier': 'Mount Gambier',
             'PearceRAAF': 'Pearce RAAF', 'SalmonGums': 'Salmon Gums',
             'AliceSprings': 'Alice Springs', "MelbourneAirport": "Melbourne",
             "SydneyAirport": "Sydney", "PerthAirport": "Perth"
        }
        df['Location'] = df['Location'].replace(ciudades_corregidas)
        df = df.merge(regiones[['Location', 'region']], on='Location', how='left')
    else:
        print("   [WARNING] No se encontró ciudades_regiones.csv en el directorio actual. Las regiones quedarán vacías.")
        df['region'] = np.nan
    
    # Creamos nuevas columnas
    df['Date'] = pd.to_datetime(df['Date'])
    df['month'] = df['Date'].dt.month
    df['Month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['Month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    def get_season(m):
        if m in [12, 1, 2]: return "summer"
        if m in [3, 4, 5]: return "autumn"
        if m in [6, 7, 8]: return "winter"
        return "spring"
    
    df['season'] = df['month'].apply(get_season)

    # Codificamos las variables cualitativas
    for s in ['spring', 'summer', 'winter']:
        df[f'season_{s}'] = (df['season'] == s).astype(float)
        
    posibles_regiones = [
        'East Coast', 'Monsoonal North', 'Murray Basin', 'Rangelands', 
        'Southern Slopes', 'Southern and South Western Flatlands', 'Wet Tropics'
    ]
    for reg in posibles_regiones:
        col_name = 'region_' + reg.replace(' ', '_')
        df[col_name] = (df['region'] == reg).astype(float)

    # 4. RainToday
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    mask_null_raintoday = df['RainToday'].isna()
    df.loc[mask_null_raintoday, 'RainToday'] = df.loc[mask_null_raintoday, 'Rainfall'].apply(lambda x: 1 if x > 1 else 0)

    # Realizamos la imputación 
    cols_num = [
        "MinTemp", "MaxTemp", "Temp9am", "Rainfall", "Temp3pm", "Pressure9am",
        "Pressure3pm", "Humidity9am", "Humidity3pm", "WindGustSpeed", "Sunshine", 
        "Evaporation", "WindSpeed9am", "WindSpeed3pm"
    ]
    for col in cols_num:
        if col in df.columns:
            val = df[col].mean() if not df[col].isna().all() else 0
            df[col] = df[col].fillna(val)

    # Viento Cíclico
    wind_cols = ["WindGustDir", "WindDir9am", "WindDir3pm"]
    wind_dir_map = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135, 
        'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270, 
        'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    for col in wind_cols:
        if col in df.columns:
            rad = df[col].map(wind_dir_map).fillna(0) 
            df[col + '_sin'] = np.sin(np.deg2rad(rad))
            df[col + '_cos'] = np.cos(np.deg2rad(rad))

    # Creamos nuevas columnas derivadas
    df['TempRange'] = df['MaxTemp'] - df['MinTemp']
    df['HumidityDiff'] = df['Humidity9am'] - df['Humidity3pm']
    df['PressureDiff'] = df['Pressure9am'] - df['Pressure3pm']
    
    df = df.fillna(0)

    # 8. Selección final
    for col in COLS_FINALES:
        if col not in df.columns:
            df[col] = 0.0
            
    return df[COLS_FINALES]

def predict():
    # Detectamos dónde está este archivo .py
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()
        if os.path.exists(os.path.join(BASE_DIR, 'docker')):
            BASE_DIR = os.path.join(BASE_DIR, 'docker')

    print(f"Directorio de trabajo establecido en: {BASE_DIR}")
    
    # Cambiamos el dir de trabajo a el dir del script para leer el Inpup File
    os.chdir(BASE_DIR)

    # Nombres de los archivos 
    MODEL_FILE = 'modelo_red_neuronal.keras'
    SCALER_FILE = 'scaler.joblib'
    INPUT_FILE = 'weatherAUS.csv' 
    OUTPUT_FILE = 'predicciones.csv'

    # Validaciones
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: No se encuentra '{MODEL_FILE}' en {BASE_DIR}")
        return
    if not os.path.exists(SCALER_FILE):
        print(f"ERROR: No se encuentra '{SCALER_FILE}' en {BASE_DIR}")
        return
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: No se encuentra '{INPUT_FILE}' en {BASE_DIR}")
        return

    try:
        print(f"1. Cargando modelo y scaler")
        model = tf.keras.models.load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)

        # Leemos el CSV 
        print(f"2. Leyendo datos desde '{INPUT_FILE}'")
        df_raw = pd.read_csv(INPUT_FILE, nrows=1) 

        # Realizamos el preprocesamiento
        print("3. Ejecutando pipeline de preprocesamiento manual")
        X_clean = preprocesar_datos(df_raw) 
        
        # Escalamos los datos
        print("4. Escalando datos")
        X_scaled = scaler.transform(X_clean)
        
        # Predecimos
        print("5. Realizando predicción\n")
        prob = model.predict(X_scaled, verbose=0).flatten()
        pred = (prob > 0.5).astype(int)
        print(f"Predicción de la primera fila:\nProbabilidad: {prob[0]:.4f} → Predicción: {pred[0]}")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    predict()