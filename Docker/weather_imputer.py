from sklearn.base import BaseEstimator, TransformerMixin

class WeatherImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mode_map_loc = {}
        self.median_map_region = {}
        self.mode_map_region = {}
        self.global_medians = {}
        self.global_mode_disc = {}
        self.global_mode_reg = {}

        self.cols_median = [
            "MinTemp","MaxTemp","Temp9am","Rainfall","Temp3pm","Pressure9am",
            "Pressure3pm","Humidity9am","Humidity3pm","WindGustSpeed","Sunshine",
            "Evaporation"
        ]
        self.cols_mode_disc = [
            "WindSpeed3pm","WindSpeed9am","WindGustDir","WindDir9am","WindDir3pm"
        ]
        self.cols_mode_reg = ["Cloud9am","Cloud3pm"]

    def fit(self, X):
        for col in self.cols_mode_disc:
            if col in X.columns:
                self.mode_map_loc[col] = X.groupby('Location')[col].apply(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
                ).to_dict()

        if 'region' in X.columns:
            self.median_map_region = X.groupby('region')[self.cols_median].median().to_dict('index')
            
            for col in self.cols_mode_reg:
                self.mode_map_region[col] = X.groupby('region')[col].apply(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
                ).to_dict()

        self.global_medians = X[self.cols_median].median().to_dict()
        self.global_mode_disc = X[self.cols_mode_disc].mode().iloc[0].to_dict()
        self.global_mode_reg = X[self.cols_mode_reg].mode().iloc[0].to_dict()

        return self

    def transform(self, X):
        X = X.copy()
        
        if 'RainToday' in X.columns and 'Rainfall' in X.columns:
            X['RainToday'] = X['RainToday'].fillna(
                X['Rainfall'].apply(lambda x: 1 if x > 0.01 else 0)
            )

        for col in self.cols_mode_disc:
            if col in X.columns:
                X[col] = X[col].fillna(X['Location'].map(self.mode_map_loc.get(col, {})))

        if 'region' in X.columns:
            for col in self.cols_median:
                if col in X.columns:
                    mask = X[col].isna()
                    if mask.any():
                        X.loc[mask, col] = X.loc[mask].apply(
                            lambda r: self.median_map_region.get(r['region'], {}).get(col, np.nan),
                            axis=1
                        )

            for col in self.cols_mode_reg:
                if col in X.columns:
                    X[col] = X[col].fillna(X['region'].map(self.mode_map_region.get(col, {})))

        X.fillna(self.global_medians, inplace=True)
        X.fillna(self.global_mode_disc, inplace=True)
        X.fillna(self.global_mode_reg, inplace=True)

        if 'MaxTemp' in X.columns and 'MinTemp' in X.columns:
            X['TempRange'] = X['MaxTemp'] - X['MinTemp']

        if 'Humidity9am' in X.columns and 'Humidity3pm' in X.columns:
            X['HumidityDiff'] = X['Humidity9am'] - X['Humidity3pm']

        if 'Pressure9am' in X.columns and 'Pressure3pm' in X.columns:
            X['PressureDiff'] = X['Pressure9am'] - X['Pressure3pm']

        return X

