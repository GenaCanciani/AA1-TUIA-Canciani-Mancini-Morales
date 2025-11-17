# ========================================================
# 0) IMPORTS
# ========================================================

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ========================================================
# 1) CUSTOM TRANSFORMERS
# ========================================================

# --------------------------------------------------------
# Añadir Región (desde CSV)
# --------------------------------------------------------
class AddRegion(BaseEstimator, TransformerMixin):
    def __init__(self, df_ciudades_regiones):
        self.df_ciudades_regiones = df_ciudades_regiones

        # Corrige nombres antes del merge
        self.ciudades_corregidas = {
            'BadgerysCreek': 'Badgerys Creek',
            'CoffsHarbour': 'Coffs Harbour',
            'NorahHead': 'Norah Head',
            'NorfolkIsland': 'Norfolk Island',
            'WaggaWagga': 'Wagga Wagga',
            'MountGinini': 'Mount Ginini',
            'Brisbane': 'Brisbane',
            'MountGambier': 'Mount Gambier',
            'PearceRAAF': 'Pearce RAAF',
            'SalmonGums': 'Salmon Gums',
            'AliceSprings': 'Alice Springs',
            "MelbourneAirport": "Melbourne",
            "SydneyAirport": "Sydney",
            "PerthAirport": "Perth"
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Aplicar corrección
        df['Location'] = df['Location'].replace(self.ciudades_corregidas)

        # Preparar tabla regiones
        df_regiones = self.df_ciudades_regiones[['Location', 'label']]\
            .rename(columns={'label': 'region'})

        # Merge seguro
        df = df.merge(df_regiones, on='Location', how='left')

        return df


# --------------------------------------------------------
# Features de fecha
# --------------------------------------------------------
class AddDateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['month'] = df['Date'].dt.month
        df['Month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df


# --------------------------------------------------------
# Estaciones
# --------------------------------------------------------
class AddSeason(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        def season(m):
            if m in [12, 1, 2]: return "summer"
            if m in [3, 4, 5]: return "autumn"
            if m in [6, 7, 8]: return "winter"
            return "spring"

        df['season'] = df['month'].apply(season)
        return df


# --------------------------------------------------------
# Direcciones de viento a codificación cíclica
# --------------------------------------------------------
class WindDirToCyclic(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.mapping = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.columns:
            rad = df[col].map(self.mapping)
            df[col + "_sin"] = np.sin(np.deg2rad(rad))
            df[col + "_cos"] = np.cos(np.deg2rad(rad))
        return df


# ========================================================
# 2) IMPUTADOR TOTAL
# ========================================================
class ImputadorTotal(BaseEstimator, TransformerMixin):

    def __init__(self, cols_median, cols_mode_disc, cols_mode_reg):
        self.cols_median = cols_median
        self.cols_mode_disc = cols_mode_disc
        self.cols_mode_reg = cols_mode_reg

    def fit(self, X, y=None):
        df = X.copy()

        # ------------ WindDir9am -----------
        self.wd9_map_date_loc = df.groupby(['Location', 'Date'])['WindDir9am']\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        self.wd9_map_loc = df.groupby('Location')['WindDir9am']\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

        # ------------ WindDir3pm -----------
        self.wd3_map_date_loc = df.groupby(['Location', 'Date'])['WindDir3pm']\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        self.wd3_map_loc = df.groupby('Location')['WindDir3pm']\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

        # ------------ Medianas region + Date -----------
        self.median_map = df.groupby(['region', 'Date'])[self.cols_median].median()

        # ------------ Modas disc (Location + Date) -----------
        self.mode_map_disc = df.groupby(['Location', 'Date'])[self.cols_mode_disc]\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

        # ------------ Modas reg (region + Date) -----------
        self.mode_map_region = df.groupby(['region', 'Date'])[self.cols_mode_reg]\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

        # ------------ Fallback Global -----------
        self.global_medians = df[self.cols_median].median()
        self.global_mode_disc = df[self.cols_mode_disc].mode().iloc[0]
        self.global_mode_reg = df[self.cols_mode_reg].mode().iloc[0]

        return self

    def transform(self, X):
        df = X.copy()

        # -------------------------------------
        # WINDDIR9AM
        # -------------------------------------
        df.set_index(['Location', 'Date'], inplace=True)
        df['WindDir9am'] = df['WindDir9am'].fillna(self.wd9_map_date_loc)
        df.reset_index(inplace=True)

        df.set_index('Location', inplace=True)
        df['WindDir9am'] = df['WindDir9am'].fillna(self.wd9_map_loc)
        df.reset_index(inplace=True)

        # -------------------------------------
        # WINDDIR3PM
        # -------------------------------------
        df.set_index(['Location', 'Date'], inplace=True)
        df['WindDir3pm'] = df['WindDir3pm'].fillna(self.wd3_map_date_loc)
        df.reset_index(inplace=True)

        df.set_index('Location', inplace=True)
        df['WindDir3pm'] = df['WindDir3pm'].fillna(self.wd3_map_loc)
        df.reset_index(inplace=True)

        # -------------------------------------
        # MEDIANAS region + Date
        # -------------------------------------
        df.set_index(['region', 'Date'], inplace=True)
        for col in self.cols_median:
            df[col] = df[col].fillna(self.median_map[col])
        df.reset_index(inplace=True)

        # -------------------------------------
        # MODA DISC (Location + Date)
        # -------------------------------------
        df.set_index(['Location', 'Date'], inplace=True)
        for col in self.cols_mode_disc:
            df[col] = df[col].fillna(self.mode_map_disc[col])
        df.reset_index(inplace=True)

        # -------------------------------------
        # MODA REG (region + Date)
        # -------------------------------------
        df.set_index(['region', 'Date'], inplace=True)
        for col in self.cols_mode_reg:
            df[col] = df[col].fillna(self.mode_map_region[col])
        df.reset_index(inplace=True)

        # -------------------------------------
        # FALLBACK GLOBAL
        # -------------------------------------
        df[self.cols_median] = df[self.cols_median].fillna(self.global_medians)
        df[self.cols_mode_disc] = df[self.cols_mode_disc].fillna(self.global_mode_disc)
        df[self.cols_mode_reg] = df[self.cols_mode_reg].fillna(self.global_mode_reg)

        # -------------------------------------
        # RainToday especial
        # -------------------------------------
        df['RainToday'] = df['RainToday'].fillna(
            df['Rainfall'].apply(lambda x: 1 if x > 0.01 else 0)
        )

        return df
    
class AddDerivedFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df['TempRange'] = df['MaxTemp'] - df['MinTemp']
        df['HumidityDiff'] = df['Humidity9am'] - df['Humidity3pm']
        df['PressureDiff'] = df['Pressure9am'] - df['Pressure3pm']

        return df


# ========================================================
# 3) PIPELINE FINAL (DEVUELVE SOLO EL DATAFRAME LISTO)
# ========================================================

def pipeline_preparacion(df, df_ciudades_regiones):

    cols_median = ["MinTemp", "MaxTemp", "Temp9am","Rainfall","Temp3pm",
                   "Pressure9am","Pressure3pm","Humidity9am","Humidity3pm",
                   "WindGustSpeed","Sunshine","Evaporation"]

    cols_mode_disc = ["WindSpeed3pm","WindSpeed9am","WindGustDir"]
    cols_mode_reg  = ["Cloud9am","Cloud3pm"]

    pipe = Pipeline([
        ("add_region", AddRegion(df_ciudades_regiones)),
        ("add_date", AddDateFeatures()),
        ("imputacion", ImputadorTotal(cols_median, cols_mode_disc, cols_mode_reg)),
        ("season", AddSeason()),
        ("derived", AddDerivedFeatures()),  
        ("wind", WindDirToCyclic(["WindGustDir", "WindDir9am", "WindDir3pm"])),
    ])

    df_final = pipe.fit_transform(df)

    return df_final


# ========================================================
# 4) EJECUCIÓN DIRECTA
# ========================================================

df_ciudades_regiones = pd.read_csv("ciudades_regiones.csv")
df = pd.read_csv("weatherAUS.csv")

df= df.dropna(subset=["RainTomorrow"])

X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

df_listo = pipeline_preparacion(X_train, df_ciudades_regiones)

df_listo.info()
print("Dataset listo sin modelo.")









































































# ========================================================
# 0) IMPORTS
# ========================================================

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ========================================================
# 1) CUSTOM TRANSFORMERS
# ========================================================

# --------------------------------------------------------
# Añadir Región (desde CSV)
# --------------------------------------------------------
class AddRegion(BaseEstimator, TransformerMixin):
    def __init__(self, df_ciudades_regiones):
        self.df_ciudades_regiones = df_ciudades_regiones

        # Corrige nombres antes del merge
        self.ciudades_corregidas = {
            'BadgerysCreek': 'Badgerys Creek',
            'CoffsHarbour': 'Coffs Harbour',
            'NorahHead': 'Norah Head',
            'NorfolkIsland': 'Norfolk Island',
            'WaggaWagga': 'Wagga Wagga',
            'MountGinini': 'Mount Ginini',
            'Brisbane': 'Brisbane',
            'MountGambier': 'Mount Gambier',
            'PearceRAAF': 'Pearce RAAF',
            'SalmonGums': 'Salmon Gums',
            'AliceSprings': 'Alice Springs',
            "MelbourneAirport": "Melbourne",
            "SydneyAirport": "Sydney",
            "PerthAirport": "Perth"
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df['Location'] = df['Location'].replace(self.ciudades_corregidas)

        df_regiones = self.df_ciudades_regiones[['Location', 'label']]\
            .rename(columns={'label': 'region'})

        df = df.merge(df_regiones, on='Location', how='left')

        return df


# --------------------------------------------------------
# Features de fecha
# --------------------------------------------------------
class AddDateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['month'] = df['Date'].dt.month
        df['Month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df


# --------------------------------------------------------
# Estaciones
# --------------------------------------------------------
class AddSeason(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        def season(m):
            if m in [12, 1, 2]: return "summer"
            if m in [3, 4, 5]: return "autumn"
            if m in [6, 7, 8]: return "winter"
            return "spring"

        df['season'] = df['month'].apply(season)
        return df


# --------------------------------------------------------
# Direcciones de viento a codificación cíclica
# --------------------------------------------------------
class WindDirToCyclic(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.mapping = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.columns:
            rad = df[col].map(self.mapping)
            df[col + "_sin"] = np.sin(np.deg2rad(rad))
            df[col + "_cos"] = np.cos(np.deg2rad(rad))
        return df


# ========================================================
# 2) IMPUTADOR TOTAL
# ========================================================
class ImputadorTotal(BaseEstimator, TransformerMixin):

    def __init__(self, cols_median, cols_mode_disc, cols_mode_reg):
        self.cols_median = cols_median
        self.cols_mode_disc = cols_mode_disc
        self.cols_mode_reg = cols_mode_reg

    def fit(self, X, y=None):
        df = X.copy()

        # ------------ WindDir9am -----------
        self.wd9_map_date_loc = df.groupby(['Location', 'Date'])['WindDir9am']\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        self.wd9_map_loc = df.groupby('Location')['WindDir9am']\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

        # ------------ WindDir3pm -----------
        self.wd3_map_date_loc = df.groupby(['Location', 'Date'])['WindDir3pm']\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        self.wd3_map_loc = df.groupby('Location')['WindDir3pm']\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

        # ------------ Medianas region + Date -----------
        self.median_map = df.groupby(['region', 'Date'])[self.cols_median].median()

        # ------------ Modas disc -----------
        self.mode_map_disc = df.groupby(['Location', 'Date'])[self.cols_mode_disc]\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

        # ------------ Modas reg -----------
        self.mode_map_region = df.groupby(['region', 'Date'])[self.cols_mode_reg]\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

        # ------------ Fallback Global -----------
        self.global_medians = df[self.cols_median].median()
        self.global_mode_disc = df[self.cols_mode_disc].mode().iloc[0]
        self.global_mode_reg = df[self.cols_mode_reg].mode().iloc[0]

        return self

    def transform(self, X):
        df = X.copy()

        # -------------------------------------
        # WINDDIR9AM
        # -------------------------------------
        df.set_index(['Location', 'Date'], inplace=True)
        df['WindDir9am'] = df['WindDir9am'].fillna(self.wd9_map_date_loc)
        df.reset_index(inplace=True)

        df.set_index('Location', inplace=True)
        df['WindDir9am'] = df['WindDir9am'].fillna(self.wd9_map_loc)
        df.reset_index(inplace=True)

        # -------------------------------------
        # WINDDIR3PM
        # -------------------------------------
        df.set_index(['Location', 'Date'], inplace=True)
        df['WindDir3pm'] = df['WindDir3pm'].fillna(self.wd3_map_date_loc)
        df.reset_index(inplace=True)

        df.set_index('Location', inplace=True)
        df['WindDir3pm'] = df['WindDir3pm'].fillna(self.wd3_map_loc)
        df.reset_index(inplace=True)

        # -------------------------------------
        # MEDIANAS region + Date
        # -------------------------------------
        df.set_index(['region', 'Date'], inplace=True)
        for col in self.cols_median:
            df[col] = df[col].fillna(self.median_map[col])
        df.reset_index(inplace=True)

        # -------------------------------------
        # MODA DISC
        # -------------------------------------
        df.set_index(['Location', 'Date'], inplace=True)
        for col in self.cols_mode_disc:
            df[col] = df[col].fillna(self.mode_map_disc[col])
        df.reset_index(inplace=True)

        # -------------------------------------
        # MODA REG
        # -------------------------------------
        df.set_index(['region', 'Date'], inplace=True)
        for col in self.cols_mode_reg:
            df[col] = df[col].fillna(self.mode_map_region[col])
        df.reset_index(inplace=True)

        # -------------------------------------
        # FALLBACK GLOBAL
        # -------------------------------------
        df[self.cols_median] = df[self.cols_median].fillna(self.global_medians)
        df[self.cols_mode_disc] = df[self.cols_mode_disc].fillna(self.global_mode_disc)
        df[self.cols_mode_reg] = df[self.cols_mode_reg].fillna(self.global_mode_reg)

        # -------------------------------------
        # RainToday Especial
        # -------------------------------------
        df['RainToday'] = df['RainToday'].fillna(
            df['Rainfall'].apply(lambda x: 1 if x > 0.01 else 0)
        )

        return df


# --------------------------------------------------------
# EXTRA: Nuevos features derivados
# --------------------------------------------------------
class AddDerivedFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['TempRange'] = df['MaxTemp'] - df['MinTemp']
        df['HumidityDiff'] = df['Humidity9am'] - df['Humidity3pm']
        df['PressureDiff'] = df['Pressure9am'] - df['Pressure3pm']
        return df


# ========================================================
# 3) PIPELINE FINAL (solo transforma)
# ========================================================

def build_pipeline(df_ciudades_regiones):

    cols_median = ["MinTemp", "MaxTemp", "Temp9am","Rainfall","Temp3pm",
                   "Pressure9am","Pressure3pm","Humidity9am","Humidity3pm",
                   "WindGustSpeed","Sunshine","Evaporation"]

    cols_mode_disc = ["WindSpeed3pm","WindSpeed9am","WindGustDir"]
    cols_mode_reg  = ["Cloud9am","Cloud3pm"]

    pipe = Pipeline([
        ("add_region", AddRegion(df_ciudades_regiones)),
        ("add_date", AddDateFeatures()),
        ("imputacion", ImputadorTotal(cols_median, cols_mode_disc, cols_mode_reg)),
        ("season", AddSeason()),
        ("derived", AddDerivedFeatures()),
        ("wind", WindDirToCyclic(["WindGustDir", "WindDir9am", "WindDir3pm"])),
    ])

    return pipe


# ========================================================
# 4) FIT TRANSFORM TRAIN / TRANSFORM TEST
# ========================================================

def pipeline_preparacion_train_test(train_df, test_df, df_ciudades_regiones):

    pipe = build_pipeline(df_ciudades_regiones)

    df_train_final = pipe.fit_transform(train_df)
    df_test_final  = pipe.transform(test_df)

    return df_train_final, df_test_final, pipe


# ========================================================
# 5) EJECUCIÓN EJEMPLO
# ========================================================

df_ciudades_regiones = pd.read_csv("ciudades_regiones.csv")
df = pd.read_csv("weatherAUS.csv")

train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

df_train_ready, df_test_ready, pipe_entrenado = pipeline_preparacion_train_test(
    train_df,
    test_df,
    df_ciudades_regiones
)

print(df_train_ready.shape, df_test_ready.shape)
print("Pipeline funcionando para train y test.")
df_train_ready.info()
df_test_ready.info()








































































# ========================================================
# 0) IMPORTS
# ========================================================

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# ========================================================
# 1) CUSTOM TRANSFORMERS
# ========================================================

# --------------------------------------------------------
# Añadir Región (desde CSV)
# --------------------------------------------------------
class AddRegion(BaseEstimator, TransformerMixin):
    def __init__(self, df_ciudades_regiones):
        self.df_ciudades_regiones = df_ciudades_regiones

        self.ciudades_corregidas = {
            'BadgerysCreek': 'Badgerys Creek',
            'CoffsHarbour': 'Coffs Harbour',
            'NorahHead': 'Norah Head',
            'NorfolkIsland': 'Norfolk Island',
            'WaggaWagga': 'Wagga Wagga',
            'MountGinini': 'Mount Ginini',
            'Brisbane': 'Brisbane',
            'MountGambier': 'Mount Gambier',
            'PearceRAAF': 'Pearce RAAF',
            'SalmonGums': 'Salmon Gums',
            'AliceSprings': 'Alice Springs',
            "MelbourneAirport": "Melbourne",
            "SydneyAirport": "Sydney",
            "PerthAirport": "Perth"
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['Location'] = df['Location'].replace(self.ciudades_corregidas)

        df_regiones = self.df_ciudades_regiones[['Location', 'label']]\
            .rename(columns={'label': 'region'})

        df = df.merge(df_regiones, on='Location', how='left')
        return df


# --------------------------------------------------------
# Features de fecha
# --------------------------------------------------------
class AddDateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['month'] = df['Date'].dt.month
        df['Month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df


# --------------------------------------------------------
# Estaciones
# --------------------------------------------------------
class AddSeason(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        def season(m):
            if m in [12, 1, 2]: return "summer"
            if m in [3, 4, 5]: return "autumn"
            if m in [6, 7, 8]: return "winter"
            return "spring"

        df['season'] = df['month'].apply(season)
        return df


# --------------------------------------------------------
# Direcciones de viento a codificación cíclica
# --------------------------------------------------------
class WindDirToCyclic(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.mapping = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.columns:
            rad = df[col].map(self.mapping)
            df[col + "_sin"] = np.sin(np.deg2rad(rad))
            df[col + "_cos"] = np.cos(np.deg2rad(rad))
        return df


# ========================================================
# 2) IMPUTADOR TOTAL
# ========================================================
class ImputadorTotal(BaseEstimator, TransformerMixin):

    def __init__(self, cols_median, cols_mode_disc, cols_mode_reg):
        self.cols_median = cols_median
        self.cols_mode_disc = cols_mode_disc
        self.cols_mode_reg = cols_mode_reg

    def fit(self, X, y=None):
        df = X.copy()

        # ------------ WindDir9am -----------
        self.wd9_map_date_loc = df.groupby(['Location', 'Date'])['WindDir9am']\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        self.wd9_map_loc = df.groupby('Location')['WindDir9am']\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

        # ------------ WindDir3pm -----------
        self.wd3_map_date_loc = df.groupby(['Location', 'Date'])['WindDir3pm']\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        self.wd3_map_loc = df.groupby('Location')['WindDir3pm']\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

        # ------------ Medianas region + Date -----------
        self.median_map = df.groupby(['region', 'Date'])[self.cols_median].median()

        # ------------ Modas disc -----------
        self.mode_map_disc = df.groupby(['Location', 'Date'])[self.cols_mode_disc]\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

        # ------------ Modas reg -----------
        self.mode_map_region = df.groupby(['region', 'Date'])[self.cols_mode_reg]\
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

        # ------------ Fallback Global -----------
        self.global_medians = df[self.cols_median].median()
        self.global_mode_disc = df[self.cols_mode_disc].mode().iloc[0]
        self.global_mode_reg = df[self.cols_mode_reg].mode().iloc[0]

        return self

    def transform(self, X):
        df = X.copy()

        # WINDDIR9AM
        df.set_index(['Location', 'Date'], inplace=True)
        df['WindDir9am'] = df['WindDir9am'].fillna(self.wd9_map_date_loc)
        df.reset_index(inplace=True)

        df.set_index('Location', inplace=True)
        df['WindDir9am'] = df['WindDir9am'].fillna(self.wd9_map_loc)
        df.reset_index(inplace=True)

        # WINDDIR3PM
        df.set_index(['Location', 'Date'], inplace=True)
        df['WindDir3pm'] = df['WindDir3pm'].fillna(self.wd3_map_date_loc)
        df.reset_index(inplace=True)

        df.set_index('Location', inplace=True)
        df['WindDir3pm'] = df['WindDir3pm'].fillna(self.wd3_map_loc)
        df.reset_index(inplace=True)

        # MEDIANAS region + Date
        df.set_index(['region', 'Date'], inplace=True)
        for col in self.cols_median:
            df[col] = df[col].fillna(self.median_map[col])
        df.reset_index(inplace=True)

        # MODA DISC
        df.set_index(['Location', 'Date'], inplace=True)
        for col in self.cols_mode_disc:
            df[col] = df[col].fillna(self.mode_map_disc[col])
        df.reset_index(inplace=True)

        # MODA REG
        df.set_index(['region', 'Date'], inplace=True)
        for col in self.cols_mode_reg:
            df[col] = df[col].fillna(self.mode_map_region[col])
        df.reset_index(inplace=True)

        # FALLBACK GLOBAL
        df[self.cols_median] = df[self.cols_median].fillna(self.global_medians)
        df[self.cols_mode_disc] = df[self.cols_mode_disc].fillna(self.global_mode_disc)
        df[self.cols_mode_reg] = df[self.cols_mode_reg].fillna(self.global_mode_reg)

        # RainToday Especial
        df['RainToday'] = df['RainToday'].fillna(
            df['Rainfall'].apply(lambda x: 1 if x > 0.01 else 0)
        )

        return df


# --------------------------------------------------------
# EXTRA: Nuevos features derivados
# --------------------------------------------------------
class AddDerivedFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['TempRange'] = df['MaxTemp'] - df['MinTemp']
        df['HumidityDiff'] = df['Humidity9am'] - df['Humidity3pm']
        df['PressureDiff'] = df['Pressure9am'] - df['Pressure3pm']
        return df


# ========================================================
# 3) PIPELINE FINAL (solo transforma)
# ========================================================

def build_pipeline(df_ciudades_regiones):

    cols_median = ["MinTemp", "MaxTemp", "Temp9am","Rainfall","Temp3pm",
                   "Pressure9am","Pressure3pm","Humidity9am","Humidity3pm",
                   "WindGustSpeed","Sunshine","Evaporation"]

    cols_mode_disc = ["WindSpeed3pm","WindSpeed9am","WindGustDir"]
    cols_mode_reg  = ["Cloud9am","Cloud3pm"]

    pipe = Pipeline([
        ("add_region", AddRegion(df_ciudades_regiones)),
        ("add_date", AddDateFeatures()),
        ("imputacion", ImputadorTotal(cols_median, cols_mode_disc, cols_mode_reg)),
        ("season", AddSeason()),
        ("derived", AddDerivedFeatures()),
        ("wind", WindDirToCyclic(["WindGustDir", "WindDir9am", "WindDir3pm"])),
    ])

    return pipe


# ========================================================
# 4) PREPARACIÓN CON STRATIFY Y FILTRO DE NULOS
# ========================================================

def preparar_train_test_con_pipeline(df, df_ciudades_regiones):

    # 1) ELIMINAR NULOS EN LA TARGET
    df_copy = df.copy()
    df_copy = df_copy.dropna(subset=['RainTomorrow'])

    # 2) DIVISIÓN X - y
    X = df_copy.drop('RainTomorrow', axis=1)
    y = df_copy['RainTomorrow']

    # 3) TRAIN TEST SPLIT (con stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4) PIPELINE
    pipe = build_pipeline(df_ciudades_regiones)

    # 5) FIT TRANSFORM
    X_train_ready = pipe.fit_transform(X_train)
    X_test_ready  = pipe.transform(X_test)

    return X_train_ready, X_test_ready, y_train, y_test, pipe


# ========================================================
# 5) EJEMPLO DE EJECUCIÓN
# ========================================================

df_ciudades_regiones = pd.read_csv("ciudades_regiones.csv")
df = pd.read_csv("weatherAUS.csv")

X_train_ready, X_test_ready, y_train, y_test, pipe_entrenado = preparar_train_test_con_pipeline(
    df,
    df_ciudades_regiones
)

print(X_train_ready.shape, X_test_ready.shape)
print("Pipeline completo funcionando con stratify y limpieza del target.")
X_train_ready.info()
X_test_ready.info()

