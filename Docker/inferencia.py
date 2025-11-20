import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
from sklearn.metrics import classification_report

# Definir nombres de columnas finales exactas
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
    print("   -> Iniciando preprocesamiento de datos crudos...")
    df = df_raw.copy()

    # 0. Eliminar target si existe
    if 'RainTomorrow' in df.columns:
        df = df.drop(columns=['RainTomorrow'])

    # 1. Cargar y mergear regiones
    # Nota: Como en predict() hacemos os.chdir al directorio del script, esto funcionará directo.
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
    
    # 2. Fechas y Estaciones
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

    # 3. One Hot Encoding
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

    # 5. Imputación Numérica
    cols_num = [
        "MinTemp", "MaxTemp", "Temp9am", "Rainfall", "Temp3pm", "Pressure9am",
        "Pressure3pm", "Humidity9am", "Humidity3pm", "WindGustSpeed", "Sunshine", 
        "Evaporation", "WindSpeed9am", "WindSpeed3pm"
    ]
    for col in cols_num:
        if col in df.columns:
            val = df[col].mean() if not df[col].isna().all() else 0
            df[col] = df[col].fillna(val)

    # 6. Viento Cíclico
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

    # 7. Features Derivados
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
    print("--- Servicio de Inferencia: Predicción de Lluvia en Australia ---")
    
    # --- 1. GESTIÓN DE RUTAS INTELIGENTE ---
    # Detectamos dónde está este archivo .py
    try:
        # Esto funciona cuando ejecutas el archivo (F5, terminal)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Esto funciona si ejecutas en celdas interactivas
        BASE_DIR = os.getcwd()
        # Ajuste por si estás en la raíz y no en la carpeta docker
        if os.path.exists(os.path.join(BASE_DIR, 'docker')):
            BASE_DIR = os.path.join(BASE_DIR, 'docker')

    print(f"Directorio de trabajo establecido en: {BASE_DIR}")
    
    # CAMBIAMOS EL DIRECTORIO DE TRABAJO A LA CARPETA DEL SCRIPT
    # Esto es crucial para que 'preprocesar_datos' encuentre 'ciudades_regiones.csv'
    # sin necesidad de pasarle rutas absolutas.
    os.chdir(BASE_DIR)

    # Nombres de archivo (ahora relativos, ya que estamos en la carpeta correcta)
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
        # 1. Cargar Artefactos
        print(f"1. Cargando modelo y scaler...")
        model = tf.keras.models.load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)

        # 2. Leer CSV Crudo
        print(f"2. Leyendo datos desde '{INPUT_FILE}'...")
        df_raw = pd.read_csv(INPUT_FILE) 
        print(f"   Registros leídos: {len(df_raw)}")

        # 3. Preprocesamiento
        print("3. Ejecutando pipeline de preprocesamiento manual...")
        # La función ahora está definida arriba, por lo que no dará error.
        X_clean = preprocesar_datos(df_raw) 
        
        # 4. Escalado
        print("4. Escalando datos...")
        X_scaled = scaler.transform(X_clean)
        
        # 5. Inferencia
        print("5. Realizando predicciones...")
        probs = model.predict(X_scaled, verbose=0).flatten()
        preds = (probs > 0.5).astype(int)
        
        # 6. Reporte de Clasificación
        if 'RainTomorrow' in df_raw.columns:
            print("\n--- Evaluación del Modelo (Classification Report) ---")
            y_true = df_raw['RainTomorrow'].map({'Yes': 1, 'No': 0})
            mask_valid = ~y_true.isna()
            
            if mask_valid.sum() > 0:
                print(classification_report(y_true[mask_valid], preds[mask_valid]))
            else:
                print("   [WARNING] No hay datos válidos en RainTomorrow.")

        # 7. Guardar
        df_raw['Prob_Lluvia'] = probs.round(4)
        df_raw['Prediccion'] = ['Yes' if x == 1 else 'No' for x in preds]
        df_raw.to_csv(OUTPUT_FILE, index=False)
        print(f"\n>>> Resultados guardados en: {os.path.abspath(OUTPUT_FILE)}")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    predict()