import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# Definir nombres de columnas finales exactas (orden importa para el Scaler)
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

    # 0. Eliminar target si existe (para simular inferencia real)
    if 'RainTomorrow' in df.columns:
        df = df.drop(columns=['RainTomorrow'])

    # 1. Cargar y mergear regiones
    # Asegúrate de que ciudades_regiones.csv esté en la carpeta docker
    if os.path.exists('ciudades_regiones.csv'):
        regiones = pd.read_csv('ciudades_regiones.csv')
        # Aseguramos nombres de columnas para el merge
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
        # Merge
        df = df.merge(regiones[['Location', 'region']], on='Location', how='left')
    else:
        print("   [WARNING] No se encontró ciudades_regiones.csv. Las regiones quedarán vacías.")
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

    # 3. One Hot Encoding Manual (Season y Region)
    # Season
    for s in ['spring', 'summer', 'winter']:
        df[f'season_{s}'] = (df['season'] == s).astype(float)
        
    # Region
    posibles_regiones = [
        'East Coast', 'Monsoonal North', 'Murray Basin', 'Rangelands', 
        'Southern Slopes', 'Southern and South Western Flatlands', 'Wet Tropics'
    ]
    for reg in posibles_regiones:
        col_name = 'region_' + reg.replace(' ', '_')
        df[col_name] = (df['region'] == reg).astype(float)

    # 4. RainToday: Mapeo e Imputación
    # Mapear Yes/No a 1/0
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    # Si es nulo, imputar basado en Rainfall > 1mm (Lógica del notebook)
    mask_null_raintoday = df['RainToday'].isna()
    df.loc[mask_null_raintoday, 'RainToday'] = df.loc[mask_null_raintoday, 'Rainfall'].apply(lambda x: 1 if x > 1 else 0)

    # 5. Imputación Numérica Simple 
    # (En producción real usaríamos valores guardados, aquí usamos la media del batch actual o 0 para no fallar)
    cols_num = [
        "MinTemp", "MaxTemp", "Temp9am", "Rainfall", "Temp3pm", "Pressure9am",
        "Pressure3pm", "Humidity9am", "Humidity3pm", "WindGustSpeed", "Sunshine", 
        "Evaporation", "WindSpeed9am", "WindSpeed3pm"
    ]
    for col in cols_num:
        if col in df.columns:
            # Rellenamos con 0 si todo es nulo, o la media si hay datos
            val = df[col].mean() if not df[col].isna().all() else 0
            df[col] = df[col].fillna(val)

    # 6. Features Cíclicos de Viento
    wind_cols = ["WindGustDir", "WindDir9am", "WindDir3pm"]
    wind_dir_map = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135, 
        'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270, 
        'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    for col in wind_cols:
        if col in df.columns:
            rad = df[col].map(wind_dir_map).fillna(0) # Asumimos Norte (0) si falta dato
            df[col + '_sin'] = np.sin(np.deg2rad(rad))
            df[col + '_cos'] = np.cos(np.deg2rad(rad))

    # 7. Features Derivados
    df['TempRange'] = df['MaxTemp'] - df['MinTemp']
    df['HumidityDiff'] = df['Humidity9am'] - df['Humidity3pm']
    df['PressureDiff'] = df['Pressure9am'] - df['Pressure3pm']
    
    # Limpieza final de nulos generados
    df = df.fillna(0)

    # 8. Selección y Ordenamiento final de columnas
    # Verificar que existan todas, sino crear en 0
    for col in COLS_FINALES:
        if col not in df.columns:
            df[col] = 0.0
            
    return df[COLS_FINALES]

def predict():
    print("--- Servicio de Inferencia: Predicción de Lluvia en Australia ---")
    
    # Archivos esperados
    MODEL_FILE = 'modelo_red_neuronal.keras'
    SCALER_FILE = 'scaler.joblib'
    INPUT_FILE = 'weatherAUS.csv' # <--- Nombre corregido
    OUTPUT_FILE = 'predicciones.csv'

    # 1. Validar Archivos
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        print("ERROR: Faltan archivos del modelo (.keras) o del scaler (.joblib).")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: No se encontró el archivo de entrada '{INPUT_FILE}'.")
        print("Por favor, asegúrese de montar el volumen o copiar el archivo.")
        return

    try:
        # 2. Cargar Artefactos
        print("1. Cargando modelo y scaler...")
        model = tf.keras.models.load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)

        # 3. Leer CSV Crudo
        print(f"2. Leyendo datos desde '{INPUT_FILE}'...")
        # Leemos solo las primeras 50 filas para probar rápido, o quita 'nrows' para todo
        # df_raw = pd.read_csv(INPUT_FILE) 
        df_raw = pd.read_csv(INPUT_FILE) 
        print(f"   Registros leídos: {len(df_raw)}")

        # 4. Preprocesamiento
        print("3. Ejecutando pipeline de preprocesamiento manual...")
        X_clean = preprocesar_datos(df_raw)
        
        # 5. Escalado
        print("4. Escalando datos...")
        X_scaled = scaler.transform(X_clean)
        
        # 6. Inferencia
        print("5. Realizando predicciones...")
        probs = model.predict(X_scaled, verbose=0).flatten()
        preds = (probs > 0.5).astype(int) # Umbral 0.5
        
        # 7. Guardar Resultados
        df_raw['Prob_Lluvia'] = probs.round(4)
        df_raw['Prediccion'] = ['Yes' if x == 1 else 'No' for x in preds]
        
        # Seleccionar columnas para mostrar
        cols_view = ['Date', 'Location', 'RainTomorrow', 'Prediccion', 'Prob_Lluvia']
        cols_existentes = [c for c in cols_view if c in df_raw.columns]
        
        print("\n--- Ejemplo de Resultados ---")
        print(df_raw[cols_existentes].head(10))
        
        df_raw.to_csv(OUTPUT_FILE, index=False)
        print(f"\n>>> Proceso finalizado. Resultados guardados en '{OUTPUT_FILE}'")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    predict()