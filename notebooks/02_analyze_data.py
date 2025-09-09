#!/usr/bin/env python3
"""
Script para análise detalhada dos dados de vendas.
"""
import os
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def load_parquet_file(file_path: Path) -> pd.DataFrame:
    """Carrega um arquivo Parquet e retorna um DataFrame."""
    try:
        logger.info(f"Carregando arquivo: {file_path.name}")
        df = pd.read_parquet(file_path)
        logger.info(f"Arquivo carregado. Dimensões: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar o arquivo {file_path.name}: {str(e)}")
        return None

def analyze_dataframe(df: pd.DataFrame, file_name: str) -> None:
    """Analisa um DataFrame e exibe informações detalhadas."""
    if df is None or df.empty:
        return
    
    print(f"\n{'='*80}")
    print(f"ANÁLISE DO ARQUIVO: {file_name}")
    print(f"{'='*80}")
    
    # Informações básicas
    print("\n1. INFORMAÇÕES GERAIS")
    print(f"Número de linhas: {len(df):,}")
    print(f"Número de colunas: {len(df.columns)}")
    
    # Mostrar colunas e tipos de dados
    print("\n2. ESTRUTURA DOS DADOS")
    print("\nColunas e tipos de dados:")
    print(df.dtypes)
    
    # Estatísticas descritivas
    print("\n3. ESTATÍSTICAS DESCRITIVAS")
    print(df.describe(include='all').T)
    
    # Verificar valores ausentes
    print("\n4. VALORES AUSENTES")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Valores Ausentes': missing,
        'Percentual (%)': missing_percent.round(2)
    })
    print(missing_df[missing_df['Valores Ausentes'] > 0])
    
    # Análise de datas (se houver coluna de data)
    date_cols = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower()]
    if date_cols:
        print("\n5. ANÁLISE TEMPORAL")
        for col in date_cols:
            try:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                
                print(f"\nPeríodo coberto por '{col}':")
                print(f"Início: {df[col].min()}")
                print(f"Fim: {df[col].max()}")
                print(f"Duração: {df[col].max() - df[col].min()}")
                
                # Plotar série temporal se for uma coluna de data
                if df[col].notna().any():
                    plt.figure(figsize=(12, 6))
                    df[col].value_counts().sort_index().plot()
                    plt.title(f'Distribuição Temporal - {col}')
                    plt.xlabel('Data')
                    plt.ylabel('Contagem')
                    plt.grid(True)
                    plt.savefig(f'temporal_analysis_{col}.png')
                    plt.close()
                    print(f"Gráfico salvo como 'temporal_analysis_{col}.png'")
            except Exception as e:
                print(f"Erro ao analisar a coluna de data '{col}': {str(e)}")
    
    # Análise de categorias (colunas com poucos valores únicos)
    print("\n6. ANÁLISE DE CATEGORIAS")
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_count = df[col].nunique()
        print(f"\nColuna: {col}")
        print(f"Valores únicos: {unique_count}")
        
        if unique_count <= 20:  # Mostrar valores se não forem muitos
            print("Valores:", df[col].value_counts().to_dict())
    
    # Verificar possíveis colunas de interesse para o modelo
    print("\n7. COLUNAS DE INTERESSE IDENTIFICADAS")
    possible_targets = [col for col in df.columns if 'venda' in col.lower() or 'quantidade' in col.lower() or 'qtd' in col.lower()]
    possible_pdv = [col for col in df.columns if 'pdv' in col.lower() or 'ponto' in col.lower() or 'loja' in col.lower()]
    possible_sku = [col for col in df.columns if 'sku' in col.lower() or 'produto' in col.lower() or 'item' in col.lower()]
    
    if possible_targets:
        print(f"Possíveis colunas alvo (target): {', '.join(possible_targets)}")
    if possible_pdv:
        print(f"Possíveis colunas de PDV: {', '.join(possible_pdv)}")
    if possible_sku:
        print(f"Possíveis colunas de SKU: {', '.join(possible_sku)}")
    
    return df

def main():
    # Criar diretório para relatórios
    os.makedirs('reports', exist_ok=True)
    
    # Diretório dos dados
    data_dir = Path("../Dados")
    
    # Listar arquivos Parquet
    parquet_files = list(data_dir.glob("*.parquet"))
    
    if not parquet_files:
        logger.error("Nenhum arquivo Parquet encontrado no diretório Dados/")
        return
    
    logger.info(f"Encontrados {len(parquet_files)} arquivos Parquet para análise")
    
    # Analisar cada arquivo
    for file_path in parquet_files:
        df = load_parquet_file(file_path)
        if df is not None:
            analyze_dataframe(df, file_path.name)
            
            # Salvar amostra dos dados
            sample_file = f'reports/sample_{file_path.stem}.csv'
            df.sample(min(1000, len(df))).to_csv(sample_file, index=False)
            logger.info(f"Amostra dos dados salva em {sample_file}")

if __name__ == "__main__":
    main()
