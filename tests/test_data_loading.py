#!/usr/bin/env python3
"""
Script de teste para verificar o carregamento dos dados
"""

import pandas as pd
import numpy as np
import os

def test_data_loading():
    print("🔍 TESTE DE CARREGAMENTO DOS DADOS")
    print("=" * 50)
    
    # Caminhos dos arquivos
    arquivos = [
        "data/raw/part-00000-tid-6364321654468257203-dc13a5d6-36ae-48c6-a018-37d8cfe34cf6-263-1-c000.snappy.parquet",
        "data/raw/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet",
        "data/raw/part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet"
    ]
    
    dataframes = []
    total_rows = 0
    
    for i, arquivo in enumerate(arquivos):
        print(f"\n📄 Testando arquivo {i+1}: {os.path.basename(arquivo)}")
        
        if os.path.exists(arquivo):
            try:
                df_temp = pd.read_parquet(arquivo)
                print(f"   ✅ Sucesso: {df_temp.shape[0]} linhas, {df_temp.shape[1]} colunas")
                
                # Mostrar algumas informações sobre o arquivo
                print(f"   📊 Colunas: {list(df_temp.columns)}")
                print(f"   🔢 Tipos de dados:")
                for col in df_temp.columns:
                    dtype = str(df_temp[col].dtype)
                    unique_count = df_temp[col].nunique()
                    null_count = df_temp[col].isnull().sum()
                    print(f"      {col}: {dtype} ({unique_count} únicos, {null_count} nulos)")
                
                # Mostrar amostra dos dados
                print(f"   📋 Primeiras 3 linhas:")
                print(df_temp.head(3))
                
                dataframes.append(df_temp)
                total_rows += df_temp.shape[0]
                
            except Exception as e:
                print(f"   ❌ Erro ao carregar: {e}")
        else:
            print(f"   ❌ Arquivo não encontrado: {arquivo}")
    
    # Tentar concatenar se há dados
    if dataframes:
        try:
            print(f"\n🔗 CONCATENAÇÃO DOS DADOS")
            print("=" * 50)
            df_combined = pd.concat(dataframes, ignore_index=True)
            print(f"✅ Dados combinados: {df_combined.shape[0]} linhas, {df_combined.shape[1]} colunas")
            
            # Estatísticas gerais
            print(f"\n📈 ESTATÍSTICAS GERAIS")
            print("=" * 50)
            print(f"Total de registros: {len(df_combined):,}")
            print(f"Colunas disponíveis: {df_combined.columns.tolist()}")
            
            # Verificar se há colunas numéricas
            numeric_cols = df_combined.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"Colunas numéricas: {list(numeric_cols)}")
                for col in numeric_cols:
                    print(f"  {col}: média={df_combined[col].mean():.2f}, "
                          f"min={df_combined[col].min()}, max={df_combined[col].max()}")
            
            return df_combined
            
        except Exception as e:
            print(f"❌ Erro na concatenação: {e}")
            return None
    else:
        print("❌ Nenhum arquivo foi carregado com sucesso!")
        return None

if __name__ == "__main__":
    resultado = test_data_loading()
    if resultado is not None:
        print(f"\n🎉 TESTE CONCLUÍDO COM SUCESSO!")
        print(f"📊 Dataset final: {resultado.shape}")
    else:
        print(f"\n❌ TESTE FALHARAM - verifique os dados e dependências")
