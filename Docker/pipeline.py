import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Agregar Región
class AddRegion(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Regiones asociadas a las ciudades
        self.region_map = {
            'Albury': 'Murray Basin', 'Badgerys Creek': 'East Coast', 'Cobar': 'Rangelands',
            'Coffs Harbour': 'East Coast', 'Moree': 'Central Slopes', 'Newcastle': 'East Coast',
            'Norah Head': 'East Coast', 'Norfolk Island': 'East Coast', 'Penrith': 'East Coast',
            'Richmond': 'East Coast', 'Sydney': 'East Coast', 'SydneyAirport': 'East Coast',
            'Wagga Wagga': 'Murray Basin', 'Williamtown': 'East Coast', 'Wollongong': 'East Coast',
            'Canberra': 'Southern Slopes', 'Tuggeranong': 'Southern Slopes', 'Mount Ginini': 'Southern Slopes',
            'Ballarat': 'Southern Slopes', 'Bendigo': 'Murray Basin', 'Sale': 'East Coast',
            'MelbourneAirport': 'Southern Slopes', 'Melbourne': 'Southern Slopes', 'Mildura': 'Murray Basin',
            'Nhil': 'Murray Basin', 'Portland': 'Southern Slopes', 'Watsonia': 'Southern Slopes',
            'Dartmoor': 'Southern Slopes', 'Brisbane': 'East Coast', 'Cairns': 'Wet Tropics',
            'GoldCoast': 'East Coast', 'Townsville': 'Monsoonal North', 'Adelaide': 'Southern and South Western Flatlands',
            'Mount Gambier': 'Southern Slopes', 'Nuriootpa': 'Southern and South Western Flatlands',
            'Woomera': 'Rangelands', 'Albany': 'Southern and South Western Flatlands', 'Witchcliffe': 'Southern and South Western Flatlands',
            'Pearce RAAF': 'Southern and South Western Flatlands', 'PerthAirport': 'Southern and South Western Flatlands',
            'Perth': 'Southern and South Western Flatlands', 'Salmon Gums': 'Southern and South Western Flatlands',
            'Walpole': 'Southern and South Western Flatlands', 'Hobart': 'Tasmania', 'Launceston': 'Tasmania',
            'Alice Springs': 'Rangelands', 'Darwin': 'Monsoonal North', 'Katherine': 'Monsoonal North', 'Uluru': 'Rangelands'
        }
        # Correcion de nombres
        self.clean_names = {
            'BadgerysCreek': 'Badgerys Creek', 'CoffsHarbour': 'Coffs Harbour', 'NorahHead': 'Norah Head',
            'NorfolkIsland': 'Norfolk Island', 'WaggaWagga': 'Wagga Wagga', 'MountGinini': 'Mount Ginini',
            'Brisbane': 'Brisbane', 'MountGambier': 'Mount Gambier', 'PearceRAAF': 'Pearce RAAF',
            'SalmonGums': 'Salmon Gums', 'AliceSprings': 'Alice Springs', "MelbourneAirport": "Melbourne",
            "SydneyAirport": "Sydney", "PerthAirport": "Perth"
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Limpiar nombres
        df['Location'] = df['Location'].replace(self.clean_names)
        # Mapear region
        df['region'] = df['Location'].map(self.region_map)
        # Rellenar desconocidos si los hubiera
        df['region'] = df['region'].fillna('East Coast')
        return df

# Features de Fecha y Estación
class DateFeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Asegurar datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['month'] = df['Date'].dt.month
            
            # Ciclo anual
            df['Month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Estaciones (One Hot Manual para garantizar columnas)
            df['season_spring'] = df['month'].apply(lambda m: 1.0 if m in [9, 10, 11] else 0.0)
            df['season_summer'] = df['month'].apply(lambda m: 1.0 if m in [12, 1, 2] else 0.0)
            df['season_winter'] = df['month'].apply(lambda m: 1.0 if m in [6, 7, 8] else 0.0)
            
            
            df = df.drop(columns=['Date', 'month'], errors='ignore')
        return df

# Imputador 
class SmartImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medians = {}
        self.modes = {}
        self.numeric_cols = [
            'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
            'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
            'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 
            'Temp9am', 'Temp3pm'
        ]
        self.cat_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

    def fit(self, X, y=None):
        # En notebook se calcula, en inferencia se aplica
        self.medians = X[self.numeric_cols].median().to_dict()
        self.modes = X[self.cat_cols].mode().iloc[0].to_dict()
        return self

    def transform(self, X):
        df = X.copy()
        # RainToday especial
        if 'RainToday' in df.columns:
            # Si es string Yes/No lo pasamos a 1/0
            df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0, 1:1, 0:0})
            # para nulos usamos rainfall
            mask = df['RainToday'].isna()
            if mask.any() and 'Rainfall' in df.columns:
                 df.loc[mask, 'RainToday'] = df.loc[mask, 'Rainfall'].apply(lambda x: 1.0 if x > 1 else 0.0)
        
        # Rellenar numéricos con medianas aprendidas
        df.fillna(value=self.medians, inplace=True)
        
        # Rellenar categóricos con modas aprendidas
        df.fillna(value=self.modes, inplace=True)
        
        return df

# Cíclica y One Hot de Region
class EncoderFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.wind_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135, 
            'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270, 
            'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        self.regiones_esperadas = [
            'region_East_Coast', 'region_Monsoonal_North', 'region_Murray_Basin',
            'region_Rangelands', 'region_Southern_Slopes',
            'region_Southern_and_South_Western_Flatlands', 'region_Wet_Tropics'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # Viento Cíclico
        for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
            if col in df.columns:
                # Mapeamos a grados
                rad = df[col].map(self.wind_map).fillna(0)
                df[col + '_sin'] = np.sin(np.deg2rad(rad))
                df[col + '_cos'] = np.cos(np.deg2rad(rad))
                df = df.drop(columns=[col])

        # One Hot Encoding de Regiones Manual
        if 'region' in df.columns:
            for col_reg in self.regiones_esperadas:
                nombre_region_limpio = col_reg.replace('region_', '').replace('_', ' ')
                df[col_reg] = (df['region'] == nombre_region_limpio).astype(float)
            df = df.drop(columns=['region', 'Location'], errors='ignore')

        # Features Derivados
        df['TempRange'] = df['MaxTemp'] - df['MinTemp']
        df['HumidityDiff'] = df['Humidity9am'] - df['Humidity3pm']
        df['PressureDiff'] = df['Pressure9am'] - df['Pressure3pm']
        
        return df

# Selector de columnas finales 
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_finales = [
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
    def fit(self, X, y=None): return self
    def transform(self, X):
        # Rellenar con 0 cualquier columna que falte y ordenar
        for col in self.cols_finales:
            if col not in X.columns:
                X[col] = 0.0
        return X[self.cols_finales]