#!/usr/bin/env python3
"""
Script para exploração inicial dos dados do desafio de previsão de vendas.
"""
import os
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_parquet_schema(file_path: str) -> None:
    """Exibe o schema de um arquivo Parquet."""
    try:
        logger.info(f"Examinando schema do arquivo: {file_path}")
        schema = pq.read_schema(file_path)
        print("\nSchema:")
        for field in schema:
            print(f"- {field.name}: {field.type}")
        
        # Mostrar amostra dos dados
        df_sample = pd.read_parquet(file_path, nrows=5)
        print("\nAmostra dos dados (5 primeiras linhas):")
        print(df_sample)
        
        # Estatísticas básicas
        print("\nInformações básicas:")
        print(f"Número de linhas: {pq.read_metadata(file_path).num_rows:,}")
        
        return df_sample
    except Exception as e:
        logger.error(f"Erro ao ler o arquivo {file_path}: {str(e)}")
        return None

def main():
    # Diretório dos dados
    data_dir = Path("../Dados")
    
    # Listar arquivos Parquet
    parquet_files = list(data_dir.glob("*.parquet"))
    
    if not parquet_files:
        logger.warning("Nenhum arquivo Parquet encontrado no diretório Dados/")
        return
    
    logger.info(f"Encontrados {len(parquet_files)} arquivos Parquet para análise")
    
    # Analisar cada arquivo
    for i, file_path in enumerate(parquet_files, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Arquivo {i}/{len(parquet_files)}: {file_path.name}")
        logger.info(f"Tamanho: {file_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Analisar schema e mostrar amostra
        df_sample = get_parquet_schema(file_path)
        
        # Se conseguimos ler o DataFrame, mostrar mais informações
        if df_sample is not None and not df_sample.empty:
            print("\nColunas disponíveis:")
            for col in df_sample.columns:
                print(f"- {col}")
            
            # Tentar inferir o conteúdo baseado nos nomes das colunas
            if 'data' in df_sample.columns or 'date' in df_sample.columns:
                print("\nPossui dados temporais")
            if 'venda' in ' '.join(df_sample.columns).lower() or 'quantidade' in ' '.join(df_sample.columns).lower():
                print("Possui dados de vendas")
            if 'pdv' in ' '.join(df_sample.columns).lower() or 'ponto_venda' in ' '.join(df_sample.columns).lower():
                print("Possui dados de PDV")
            if 'sku' in ' '.join(df_sample.columns).lower() or 'produto' in ' '.join(df_sample.columns).lower():
                print("Possui dados de SKU/produtos")

if __name__ == "__main__":
    main()
