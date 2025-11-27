import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
from weather_imputer import WeatherImputer

def predict():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()
        if os.path.exists(os.path.join(BASE_DIR, 'docker')):
            BASE_DIR = os.path.join(BASE_DIR, 'docker')

    print(f"Directorio de trabajo establecido en: {BASE_DIR}")
    
    os.chdir(BASE_DIR)

    MODEL_FILE = 'modelo_red_neuronal.keras'
    SCALER_FILE = 'scaler.joblib'
    PREPROCESSOR_FILE = 'weather_imputer.pkl'    
    INPUT_FILE = 'primer_fila.csv' 
    

    # Validaciones
    for f in [MODEL_FILE, SCALER_FILE, PREPROCESSOR_FILE, INPUT_FILE]:
        if not os.path.exists(f):
            print(f"ERROR: No se encuentra '{f}' en {BASE_DIR}")
            return

    try:
        print(f"1. Cargando modelo, scaler y preprocesador")
        model = tf.keras.models.load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        preprocessor = joblib.load(PREPROCESSOR_FILE) 

        print(f"2. Leyendo datos desde '{INPUT_FILE}'")
        df_raw = pd.read_csv(INPUT_FILE, nrows=1)

        print("3. Aplicando transform() del preprocesador")
        X_clean = preprocessor.transform(df_raw)       

        print("4. Escalando datos")
        #X_scaled = scaler.transform(X_clean)
        
        print("5. Realizando predicción\n")
        prob = model.predict(X_clean, verbose=0).flatten()
        pred = (prob > 0.5).astype(int)
        print(f"Predicción:\nProbabilidad: {prob[0]:.4f} → Predicción: {pred[0]}")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    predict()
