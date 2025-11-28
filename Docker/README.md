# Despliegue de Modelo de Predicción de Lluvia - Australia

Este directorio contiene los archivos necesarios para construir y ejecutar un contenedor Docker que realiza inferencias utilizando una **Red Neuronal Profunda** entrenada para predecir si lloverá al día siguiente en Australia.

## Contenido del Directorio

*   **`Dockerfile`**: Archivo de configuración para construir la imagen (basada en Python 3.10).
*   **`inferencia.py`**: Script principal de Python. Orquesta la carga del modelo y el pipeline, ejecuta la transformación de los datos y genera la predicción final.
*   **`pipeline.py`**: Definición de las clases y transformadores personalizados necesarios para el preprocesamiento.
*   **`pipeline_completo.pkl`**: Objeto serializado que contiene toda la lógica de limpieza, imputación y escalado aprendida durante el entrenamiento.
*   **`modelo_red_neuronal.keras`**: El modelo de Deep Learning optimizado con Optuna y entrenado.
*   **`requirements.txt`**: Lista de dependencias necesarias (TensorFlow-CPU, Pandas, Scikit-learn, Joblib).
*   **`input_ejemplo.csv`**: Dataset de entrada con datos meteorológicos crudos para probar la inferencia (puede ser una muestra o el dataset `weatherAUS.csv` completo).

## Instrucciones de Ejecución

**Requisitos previos:** Tener Docker Desktop o Docker Engine instalado y corriendo. En la terminal y desde la carpeta de Docker,
ejecuta el comando pip install -r requirements.txt para descargar las librerías necesarias.

### 1. Construir la Imagen Docker

Abre una terminal, navega hasta esta carpeta (`/docker`) y ejecuta el siguiente comando:

docker build -t prediccion-lluvia-australia .

luego:

docker run --rm prediccion-lluvia-australia
