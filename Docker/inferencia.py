import pandas as pd
import joblib
import tensorflow as tf
import os
import sys
import numpy as np

from pipeline import AddRegion, DateFeatureEngineering, SmartImputer, EncoderFeatures, ColumnSelector

def predict():
    print("--- Servicio de Inferencia ---")
    
    # rutas 
    PIPELINE_FILE = 'pipeline_completo.pkl'
    MODEL_FILE = 'modelo_red_neuronal.keras'
    INPUT_FILE = 'input_ejemplo.csv' 

    # Carga de pipeline completo
    try:
        print("Cargando pipeline y modelo...")
        pipeline = joblib.load(PIPELINE_FILE)
        model = tf.keras.models.load_model(MODEL_FILE)
    except Exception as e:
        print(f"Error cargando archivos: {e}")
        return

    # Busca el archivo
    if not os.path.exists(INPUT_FILE):
        print(f"No se encontró {INPUT_FILE}")
        return
        
    df_new = pd.read_csv(INPUT_FILE)
    
    # Borra el target si esta
    if 'RainTomorrow' in df_new.columns:
        df_new = df_new.drop(columns=['RainTomorrow'])
        
    print(f"Datos leídos: {df_new.shape}")

    # Se usan los datos aprendidos en el pipeline
    try:
        print("Procesando datos...")
        X_processed = pipeline.transform(df_new)
        
        # 5. Predicción
        print("Prediciendo...")
        probs = model.predict(X_processed, verbose=0).flatten()
        
        # Mostrar resultados
        df_new['Probabilidad'] = probs
        df_new['Prediccion'] = np.where(probs > 0.5, 'Si llueve', 'No llueve')
        print("\nRESULTADO:")
        print(df_new[['Date', 'Location', 'Probabilidad', 'Prediccion']])
        
    except Exception as e:
        print(f"Error en inferencia: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    predict()