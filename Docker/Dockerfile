FROM python:3.10-slim

# Configuración de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencias de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar Scripts y Binarios
COPY inferencia.py .
COPY scaler.joblib .
COPY modelo_red_neuronal.keras .
COPY ciudades_regiones.csv .

# Copiar dataset crudo (opcional, idealmente se monta como volumen)
COPY weatherAUS.csv . 

CMD ["python", "inferencia.py"]