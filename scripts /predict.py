import argparse
import logging
import os
import pandas as pd
from datetime import datetime
import joblib
from forecaster_class import SalesForecasterV2

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(model_path: str, data_path: str, output_path: str):
    """
    Função principal para gerar o arquivo de submissão a partir de um modelo treinado.
    """
    logging.info("Iniciando o Pipeline de Previsão (Submissão).")
    
    # Configuração de caminhos
    file_paths = {
        'vendas': os.path.join(data_path, 'raw/fato_vendas.parquet'),
        'pdvs': os.path.join(data_path, 'raw/dim_pdvs.parquet'),
        'produtos': os.path.join(data_path, 'raw/dim_produtos.parquet')
    }

    # Carrega o modelo salvo e o "injeta" em uma nova instância da classe
    try:
        artifacts = joblib.load(model_path)
        predictor = SalesForecasterV2()
        predictor.model = artifacts['model']
        predictor.feature_names = artifacts['feature_names']
        predictor.categorical_features = artifacts['categorical_features']
        logging.info("Modelo e artefatos carregados com sucesso.")
    except FileNotFoundError:
        logging.error(f"Arquivo do modelo não encontrado em '{model_path}'. Execute o train.py primeiro.")
        return

    # Lógica para gerar o arquivo com limite de linhas
    try:
        df_full_data = predictor.load_data(file_paths)
        df_historical_2022 = df_full_data[df_full_data['ano'] == 2022].copy()

        logging.info("Selecionando as Top 300.000 combinações (PDV, SKU) com base nas vendas de 2022.")
        vendas_totais_2022 = df_historical_2022.groupby(['pdv', 'sku'])['quantidade'].sum().reset_index()
        top_combinacoes = vendas_totais_2022.nlargest(300000, 'quantidade')
        df_historical_filtrado = pd.merge(df_historical_2022, top_combinacoes[['pdv', 'sku']], on=['pdv', 'sku'], how='inner')
        
        forecasts = predictor.generate_forecasts(df_historical_filtrado, weeks_to_forecast=5)

        if not forecasts.empty:
            df_submission = forecasts.rename(columns={'sku': 'produto', 'quantidade_prevista': 'quantidade'})
            df_submission = df_submission[['semana', 'pdv', 'produto', 'quantidade']]
            df_submission_sorted = df_submission.sort_values(by=['semana', 'quantidade'], ascending=[True, False])
            
            os.makedirs(output_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            submission_filename = os.path.join(output_path, f"previsao_SUBMISSAO_{timestamp}.parquet")
            df_submission_sorted.to_parquet(submission_filename, index=False)
            logging.info(f"ARQUIVO DE SUBMISSÃO salvo em: {submission_filename}")
        else:
            logging.warning("Nenhuma previsão foi gerada.")

    except Exception as e:
        logging.error(f"O pipeline de previsão falhou com o erro: {e}")
        raise e

    logging.info("Pipeline de Previsão (Submissão) finalizado com sucesso!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera o arquivo de previsão para submissão.")
    parser.add_argument("--model_path", type=str, default="artifacts/sales_forecaster_v2_final.joblib", help="Caminho para o modelo treinado.")
    parser.add_argument("--data_path", type=str, default="data", help="Caminho para a pasta 'data'.")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Caminho para salvar a previsão final.")
    
    args = parser.parse_args()
    
    main(args.model_path, args.data_path, args.output_path)
