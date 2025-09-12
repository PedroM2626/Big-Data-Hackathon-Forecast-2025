# -*- coding: utf-8 -*-
"""
Módulo de Previsão de Vendas (Forecast) - One-Click Order

Este módulo contém uma solução completa e robusta para prever a quantidade
semanal de vendas por Ponto de Venda (PDV) e SKU. O objetivo é apoiar o processo
de reposição de estoque para as primeiras semanas de 2023, com base no histórico
de vendas de 2022.

O pipeline foi reestruturado para seguir as melhores práticas de engenharia
de software e machine learning, incluindo:
- Estrutura orientada a objetos para manutenibilidade.
- Engenharia de features avançada (lags e janelas móveis).
- Utilização do LightGBM, um modelo de alta performance.
- Validação cruzada temporal e tuning de hiperparâmetros.
- Logging profissional e documentação completa.

Autor: BSB Data 01
Data da Versão: 2025-09-12
"""

import logging
import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

# ==============================================================================
# Configuração Inicial
# ==============================================================================

# Configuração de logging para substituir 'print'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configurações de exibição e estilo
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')


# ==============================================================================
# Classe Principal do Modelo de Previsão
# ==============================================================================

class SalesForecaster:
    """
    Encapsula todo o pipeline de previsão de vendas, desde o carregamento de
    dados até o treinamento, avaliação e geração de previsões.
    """

    def __init__(self):
        """Inicializa o objeto, definindo os atributos iniciais."""
        self.model = None
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.performance_metrics: Dict[str, float] = {}

    def load_data(self, file_paths: Dict[str, str]) -> pd.DataFrame:
        """
        Carrega os dados normalizados, une as tabelas usando os nomes de colunas
        reais e prepara um DataFrame padronizado para a modelagem.
        """
        logging.info("Iniciando o carregamento dos dados normalizados.")

        try:
            df_vendas = pd.read_parquet(file_paths['vendas'])
            df_pdvs = pd.read_parquet(file_paths['pdvs'])
            df_produtos = pd.read_parquet(file_paths['produtos'])
            logging.info("Arquivos de vendas, pdvs e produtos carregados com sucesso.")
        except (FileNotFoundError, KeyError) as e:
            logging.error(f"Erro ao carregar os arquivos. Verifique os caminhos no dicionário 'file_paths'. Erro: {e}")
            raise

        logging.info("Iniciando a união (merge) das tabelas.")
        df_merged = pd.merge(
            df_vendas,
            df_pdvs,
            left_on='internal_store_id',
            right_on='pdv',
            how='inner'
        )
        df_merged = pd.merge(
            df_merged,
            df_produtos,
            left_on='internal_product_id',
            right_on='produto',
            how='inner'
        )
        logging.info(f"Tabelas unidas. DataFrame resultante com {df_merged.shape[0]} registros.")

        df_merged['transaction_date'] = pd.to_datetime(df_merged['transaction_date'])
        df_merged['ano'] = df_merged['transaction_date'].dt.isocalendar().year
        df_merged['semana'] = df_merged['transaction_date'].dt.isocalendar().week

        logging.info("Agregando dados de vendas por semana/pdv/produto.")
        df_aggregated = df_merged.groupby(
            ['ano', 'semana', 'pdv', 'produto']
        ).agg(
            total_quantity=('quantity', 'sum')
        ).reset_index()

        df_aggregated.rename(columns={
            'produto': 'sku',
            'total_quantity': 'quantidade'
        }, inplace=True)
        
        logging.info(f"Dados agregados e renomeados. DataFrame final com {df_aggregated.shape[0]} registros.")
        
        required_columns = ['ano', 'semana', 'pdv', 'sku', 'quantidade']
        df_aggregated.sort_values(by=required_columns, inplace=True)
        return df_aggregated[required_columns]


    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features avançadas para o modelo de machine learning.
        Esta função já espera o DataFrame com as colunas padronizadas.
        """
        logging.info("Iniciando a engenharia de features.")
        
        df_featured = df.copy()
        
        df_featured['trimestre'] = (df_featured['semana'] - 1) // 13 + 1
        df_featured['seno_semana'] = np.sin(2 * np.pi * df_featured['semana'] / 52)
        df_featured['cosseno_semana'] = np.cos(2 * np.pi * df_featured['semana'] / 52)

        df_featured.sort_values(['pdv', 'sku', 'ano', 'semana'], inplace=True)

        df_featured['lag_1_semana'] = df_featured.groupby(['pdv', 'sku'])['quantidade'].shift(1)
        df_featured['rolling_mean_4_semanas'] = df_featured.groupby(['pdv', 'sku'])['quantidade'].shift(1).rolling(window=4, min_periods=1).mean()
        df_featured['rolling_std_4_semanas'] = df_featured.groupby(['pdv', 'sku'])['quantidade'].shift(1).rolling(window=4, min_periods=1).std()

        df_featured.fillna(0, inplace=True)
        
        logging.info("Engenharia de features concluída.")
        return df_featured

    def _prepare_data_for_model(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara os dados para o treinamento, codificando 'pdv' e 'sku'.
        """
        logging.info("Preparando dados para modelagem (codificação e seleção).")
        df_model = df.copy()

        for col in ['pdv', 'sku']:
            encoder = LabelEncoder()
            # Certifica-se de que os dados são strings para o encoder
            df_model[f'{col}_encoded'] = encoder.fit_transform(df_model[col].astype(str))
            self.encoders[col] = encoder

        self.feature_names = [
            'semana', 'trimestre', 'seno_semana', 'cosseno_semana',
            'pdv_encoded', 'sku_encoded',
            'lag_1_semana', 'rolling_mean_4_semanas', 'rolling_std_4_semanas'
        ]
        
        X = df_model[self.feature_names]
        y = df_model['quantidade']

        return X, y

    def train(self, df: pd.DataFrame, tune_hyperparameters: bool = False):
        """
        CORRIGIDO: Primeiro cria as features e DEPOIS prepara os dados para o modelo.
        """
        logging.info("Iniciando o processo de treinamento do modelo.")

        # Filtra apenas os dados de 2022 para treinar
        df_train_raw = df[df['ano'] == 2022].copy()
        
        if df_train_raw.empty:
            raise ValueError("Não há dados históricos de 2022 para treinar o modelo.")

        # PASSO 1: Criar as features (trimestre, lag, etc.)
        df_train_featured = self.feature_engineering(df_train_raw)

        # PASSO 2: Preparar os dados para o modelo (codificar 'pdv' e 'sku')
        X_train, y_train = self._prepare_data_for_model(df_train_featured)

        logging.info(f"Dados de treino preparados: {len(X_train)} registros do ano de 2022.")

        if tune_hyperparameters:
            logging.info("Iniciando o tuning de hiperparâmetros com GridSearchCV e TimeSeriesSplit.")
            tscv = TimeSeriesSplit(n_splits=5)
            param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'num_leaves': [31, 50]}
            model = lgb.LGBMRegressor(random_state=42, objective='regression_l1')
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            logging.info(f"Melhores hiperparâmetros encontrados: {grid_search.best_params_}")
        else:
            logging.info("Treinando modelo LightGBM com hiperparâmetros padrão.")
            self.model = lgb.LGBMRegressor(random_state=42, objective='regression_l1', n_estimators=200, learning_rate=0.1, num_leaves=31)
            self.model.fit(X_train, y_train)
        
        logging.info("Treinamento concluído.")

    def generate_forecasts(self, df_historical: pd.DataFrame, weeks_to_forecast: int) -> pd.DataFrame:
        """
        Gera previsões para as semanas futuras de forma iterativa.
        """
        if not self.model:
            raise RuntimeError("O modelo não foi treinado. Execute o método 'train' primeiro.")

        logging.info(f"Iniciando a geração de previsões para {weeks_to_forecast} semanas.")

        forecast_df = df_historical.copy()
        all_forecasts = []

        for i in range(1, weeks_to_forecast + 1):
            current_week = i
            logging.info(f"Processando previsões para a semana {current_week} de 2023.")
            
            features_base = self.feature_engineering(forecast_df)
            latest_entries = features_base.sort_values(by=['ano', 'semana']).drop_duplicates(subset=['pdv', 'sku'], keep='last')
            
            if latest_entries.empty:
                logging.warning(f"Não há dados base para prever a semana {current_week}.")
                continue
            
            X_pred = latest_entries.copy()
            X_pred['semana'] = current_week
            X_pred['ano'] = 2023
            X_pred['trimestre'] = 1
            X_pred['seno_semana'] = np.sin(2 * np.pi * current_week / 52)
            X_pred['cosseno_semana'] = np.cos(2 * np.pi * current_week / 52)

            for col, encoder in self.encoders.items():
                known_categories = set(encoder.classes_)
                # Converte para string para garantir a consistência
                X_pred[f'{col}_encoded'] = [encoder.transform([str(item)])[0] if str(item) in known_categories else -1 for item in X_pred[col]]

            X_pred = X_pred[~(X_pred[['pdv_encoded', 'sku_encoded']] == -1).any(axis=1)]

            if X_pred.empty:
                logging.warning(f"Nenhum PDV/SKU conhecido para prever na semana {current_week}.")
                continue

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

    def save_model(self, path: str = "artifacts/sales_forecaster.joblib"):
        """
        Salva o modelo treinado e os artefatos necessários.
        """
        if not self.model:
            raise RuntimeError("Nenhum modelo treinado para salvar.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        artifacts = {
            "model": self.model,
            "encoders": self.encoders,
            "feature_names": self.feature_names,
        }
        joblib.dump(artifacts, path)
        logging.info(f"Modelo e artefatos salvos em: '{path}'")

# ==============================================================================
# Execução do Pipeline
# ==============================================================================

def main():
    """Função principal para executar o pipeline de forecasting."""
    
    logging.info("Iniciando o Pipeline de Previsão de Vendas.")

    file_paths = {
        'pdvs': r'data/raw/dim_pdvs.parquet',
        'vendas': r'data/raw/fato_vendas.parquet',
        'produtos': r'data/raw/dim_produtos.parquet'
    }
    
    forecaster = SalesForecaster()
    try:
        df_historical = forecaster.load_data(file_paths)
    except (FileNotFoundError, ValueError, KeyError) as e:
        logging.error(f"Processo encerrado devido a um erro no carregamento de dados: {e}")
        return

    try:
        forecaster.train(df_historical, tune_hyperparameters=False)
        forecaster.save_model()

        forecasts = forecaster.generate_forecasts(df_historical, weeks_to_forecast=5)

        if not forecasts.empty:
            logging.info(f"Total de {len(forecasts)} previsões geradas para Janeiro/2023.")
            output_dir = "data/processed"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            forecast_filename = os.path.join(output_dir, f"previsoes_janeiro_2023_{timestamp}.csv")
            forecasts.to_csv(forecast_filename, index=False)
            logging.info(f"Previsões salvas em: {forecast_filename}")
        else:
            logging.warning("Nenhuma previsão foi gerada.")

    except (RuntimeError, ValueError) as e:
        logging.error(f"Ocorreu um erro durante o treinamento ou previsão: {e}")
        return

    logging.info("Pipeline de Previsão de Vendas finalizado com sucesso.")


if __name__ == "__main__":

    main()
