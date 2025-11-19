# Despliegue de Modelo de Predicción de Lluvia - Australia

Este directorio contiene los archivos necesarios para construir y ejecutar un contenedor Docker que realiza inferencias utilizando una **Red Neuronal Profunda** entrenada para predecir si lloverá al día siguiente en Australia.

## Contenido del Directorio

*   **`Dockerfile`**: Archivo de configuración para construir la imagen (basada en Python 3.10).
*   **`inferencia.py`**: Script principal de Python. Construye la carga del modelo, el preprocesamiento de datos crudos (limpieza, imputación, ingeniería de features) y la predicción.
*   **`requirements.txt`**: Lista de dependencias necesarias (TensorFlow, Pandas, Scikit-learn, etc.).
*   **`modelo_red_neuronal.keras`**: El modelo de Red Neuronal optimizado con Optuna y entrenado.
*   **`scaler.joblib`**: El objeto `StandardScaler` ajustado con los datos de entrenamiento.
*   **`ciudades_regiones.csv`**: Dataset auxiliar para mapear ciudades a regiones climáticas.
*   **`weatherAUS.csv`**: Dataset de entrada con datos meteorológicos crudos para procesar.

## Instrucciones de Ejecución

Requisitos previos: Tener Docker Desktop o Docker Engine instalado y corriendo.

###  Construir la Imagen Docker
### Abre una terminal, navega hasta esta carpeta (`/docker`) y ejecuta:

docker build -t prediccion-lluvia-australia .

### Nota: Este proceso puede tardar unos minutos la primera vez, ya que debe descargar las librerías de TensorFlow y Python.
### Ejecutar el Contenedor

docker run --rm prediccion-lluvia-australia

<!-- ¿Qué hace el script?
Carga el modelo (.keras) y el escalador (.joblib).
Lee el archivo weatherAUS.csv.
Aplica el pipeline de preprocesamiento manualmente (imputación de nulos, creación de variables cíclicas para el viento, cálculo de diferencias de presión/humedad, etc.), replicando la lógica del notebook de entrenamiento.
Realiza la predicción sobre el dataset completo.
Muestra en consola una muestra de los resultados con la probabilidad calculada.
Guarda los resultados completos en un archivo interno predicciones.csv (visible si se monta un volumen, o simplemente como log de finalización). -->