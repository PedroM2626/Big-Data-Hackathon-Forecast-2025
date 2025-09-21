import logging
import os
from typing import Dict, List, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import optuna

# Classe pública que define o que será o 'cérebro' do modelo, como ele deverá agir ao treinar.
class SalesForecasterV2:
    def __init__(self):
        self.model = None
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.performance_metrics: Dict[str, float] = {}

    # Carregar os dados para análise
    def load_data(self, file_paths: Dict[str, str]) -> pd.DataFrame:
        logging.info("Iniciando o carregamento dos dados normalizados.")
        try:
            df_vendas = pd.read_parquet(file_paths['vendas'])
            df_pdvs = pd.read_parquet(file_paths['pdvs'])
            df_produtos = pd.read_parquet(file_paths['produtos'])
        except (FileNotFoundError, KeyError) as e:
            logging.error(f"Erro ao carregar os arquivos. Erro: {e}")
            raise
        df_merged = pd.merge(df_vendas, df_pdvs, left_on='internal_store_id', right_on='pdv', how='inner')
        df_merged = pd.merge(df_merged, df_produtos, left_on='internal_product_id', right_on='produto', how='inner')
        df_merged['transaction_date'] = pd.to_datetime(df_merged['transaction_date'])
        df_merged['ano'] = df_merged['transaction_date'].dt.isocalendar().year
        df_merged['semana'] = df_merged['transaction_date'].dt.isocalendar().week
        logging.info("Agregando dados de vendas por semana/pdv/produto.")
        agg_vendas = df_merged.groupby(['ano', 'semana', 'pdv', 'produto']).agg(total_quantity=('quantity', 'sum')).reset_index()
        df_aggregated = agg_vendas.rename(columns={'produto': 'sku', 'total_quantity': 'quantidade'})
        logging.info(f"Dados agregados e enriquecidos. DataFrame final com {df_aggregated.shape[0]} registros.")
        return df_aggregated

    # Engenharia de features:
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df_featured = df.copy()
        df_featured.sort_values(['pdv', 'sku', 'ano', 'semana'], inplace=True)
        df_featured['trimestre'] = (df_featured['semana'] - 1) // 13 + 1
        df_featured['seno_semana'] = np.sin(2 * np.pi * df_featured['semana'] / 52)
        df_featured['cosseno_semana'] = np.cos(2 * np.pi * df_featured['semana'] / 52)
        lags = [1, 2, 3, 4, 12, 52]
        for lag in lags:
            df_featured[f'lag_{lag}_semanas'] = df_featured.groupby(['pdv', 'sku'])['quantidade'].shift(lag)
        windows = [4, 12, 52]
        for window in windows:
            df_featured[f'rolling_mean_{window}_semanas'] = df_featured.groupby(['pdv', 'sku'])['quantidade'].shift(1).rolling(window=window, min_periods=1).mean()
            df_featured[f'rolling_std_{window}_semanas'] = df_featured.groupby(['pdv', 'sku'])['quantidade'].shift(1).rolling(window=window, min_periods=1).std()
            df_featured[f'rolling_max_{window}_semanas'] = df_featured.groupby(['pdv', 'sku'])['quantidade'].shift(1).rolling(window=window, min_periods=1).max()
        df_featured.fillna(0, inplace=True)
        return df_featured

    # Engenharia de dados
    def _prepare_data_for_model(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df_model = df.copy()
        self.categorical_features = ['pdv', 'sku']
        for col in self.categorical_features:
            df_model[col] = df_model[col].astype('category')
        self.feature_names = [
            'semana', 'trimestre', 'seno_semana', 'cosseno_semana', 'pdv', 'sku',
            'lag_1_semanas', 'lag_2_semanas', 'lag_3_semanas', 'lag_4_semanas', 'lag_12_semanas', 'lag_52_semanas',
            'rolling_mean_4_semanas', 'rolling_std_4_semanas', 'rolling_max_4_semanas',
            'rolling_mean_12_semanas', 'rolling_std_12_semanas', 'rolling_max_12_semanas',
            'rolling_mean_52_semanas', 'rolling_std_52_semanas', 'rolling_max_52_semanas',
        ]
        X = df_model[self.feature_names]
        y = df_model['quantidade']
        return X, y

    # A ser importado: Garante o treinamento do modelo. Define número de trials, parâmetros, hiperparâmetros e etc.
    def train(self, df: pd.DataFrame, validation_split_week: int = 48, use_optuna: bool = True, n_trials: int = 100):
        df_train_raw = df[df['ano'] == 2022].copy()
        if df_train_raw.empty: raise ValueError("Não há dados históricos de 2022 para treinar o modelo.")
        df_featured = self.feature_engineering(df_train_raw)
        train_set = df_featured[df_featured['semana'] < validation_split_week]
        val_set = df_featured[df_featured['semana'] >= validation_split_week]
        X_train, y_train = self._prepare_data_for_model(train_set)
        X_val, y_val = self._prepare_data_for_model(val_set)
        for col in self.categorical_features:
            all_categories = pd.concat([X_train[col], X_val[col]]).astype('category').cat.categories
            X_train[col] = pd.Categorical(X_train[col], categories=all_categories)
            X_val[col] = pd.Categorical(X_val[col], categories=all_categories)
        fit_params = {"eval_set": [(X_val, y_val)],"eval_metric": "mae","callbacks": [lgb.early_stopping(10, verbose=False)]}
        if use_optuna:
            def objective(trial):
                params = { 'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 1000, 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3), 'num_leaves': trial.suggest_int('num_leaves', 20, 300), 'max_depth': trial.suggest_int('max_depth', 3, 12), 'min_child_samples': trial.suggest_int('min_child_samples', 5, 100), 'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0), 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0), 'bagging_freq': trial.suggest_int('bagging_freq', 1, 7), 'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True), 'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True), 'random_state': 42, 'n_jobs': -1}
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, **fit_params, categorical_feature=self.categorical_features)
                preds = model.predict(X_val)
                mae = mean_absolute_error(y_val, preds)
                return mae
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            self.model = lgb.LGBMRegressor(objective='regression_l1', random_state=42, n_estimators=1000, **study.best_params)
        else:
            self.model = lgb.LGBMRegressor(objective='regression_l1', random_state=42, n_estimators=1000)
        self.model.fit(X_train, y_train, **fit_params, categorical_feature=self.categorical_features)
        val_preds = self.model.predict(X_val)
        mae = mean_absolute_error(y_val, val_preds)
        self.performance_metrics['validation_mae'] = mae
        logging.info(f"Treinamento concluído. MAE no set de validação: {mae:.4f}")

    # A ser importado:  Faz as devidas previsões depois do treinamento bem-sucedido
    def generate_forecasts(self, df_historical: pd.DataFrame, weeks_to_forecast: int) -> pd.DataFrame:
        if not self.model: raise RuntimeError("O modelo não foi treinado.")
        forecast_df = df_historical.copy()
        all_forecasts = []
        for i in range(1, weeks_to_forecast + 1):
            current_week = i
            features_base = self.feature_engineering(forecast_df)
            latest_entries = features_base.sort_values(by=['ano', 'semana']).drop_duplicates(subset=['pdv', 'sku'], keep='last')
            if latest_entries.empty: continue
            X_pred = latest_entries.copy()
            X_pred['semana'] = current_week
            X_pred['ano'] = 2023
            for col in self.categorical_features:
                model_categories = self.model.booster_.pandas_categorical[self.categorical_features.index(col)]
                X_pred[col] = pd.Categorical(X_pred[col], categories=model_categories)
            X_pred.dropna(subset=self.categorical_features, inplace=True)
            if X_pred.empty: continue
            predictions = self.model.predict(X_pred[self.feature_names])
            predictions = np.maximum(0, np.round(predictions)).astype(int)
            week_forecast = X_pred[['pdv', 'sku']].copy()
            week_forecast['semana'] = current_week
            week_forecast['quantidade_prevista'] = predictions
            all_forecasts.append(week_forecast)
            new_data = week_forecast.rename(columns={'quantidade_prevista': 'quantidade'})
            new_data['ano'] = 2023
            forecast_df = pd.concat([forecast_df, new_data], ignore_index=True)
        return pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()

    # A ser importado: Salva o artefato (modelo baseado em ML treinado, pronto para receber comandos como prever dados futruros) na pasta dedicada.
    def save_model(self, path: str):
        if not self.model: raise RuntimeError("Nenhum modelo treinado para salvar.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        artifacts = {"model": self.model, "feature_names": self.feature_names, "categorical_features": self.categorical_features}
        joblib.dump(artifacts, path)
        logging.info(f"Modelo e artefatos V2 salvos em: '{path}'")

