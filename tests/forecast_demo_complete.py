#!/usr/bin/env python3
"""
DEMO FUNCIONAL - Modelo de Forecasting para One-Click Order
Sistema para prever quantidade semanal de vendas por PDV/SKU
Usando dados simulados realistas
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime

# Configurações
warnings.filterwarnings('ignore')
np.random.seed(42)

def main():
    print("🎯 MODELO DE FORECAST - DEMO FUNCIONAL")
    print("=" * 60)
    print("📅 META: Prever vendas para 5 semanas de janeiro/2023")
    print("📊 BASE: Dados históricos simulados de 2022")
    print("⚠️  Usando dados simulados para demonstração")
    
    # 1. CRIAÇÃO DOS DADOS HISTÓRICOS
    print("\n📂 ETAPA 1: CRIAÇÃO DOS DADOS HISTÓRICOS")
    print("-" * 50)
    
    df_historical = create_historical_data()
    print(f"✅ Dados criados: {df_historical.shape[0]:,} registros")
    
    # 2. ANÁLISE EXPLORATÓRIA
    print("\n📈 ETAPA 2: ANÁLISE EXPLORATÓRIA")
    print("-" * 50)
    
    analyze_data(df_historical)
    
    # 3. CONSTRUÇÃO DO MODELO
    print("\n🤖 ETAPA 3: CONSTRUÇÃO DO MODELO")
    print("-" * 50)
    
    model, metrics = build_model(df_historical)
    print(f"✅ Modelo treinado com sucesso!")
    print(f"📊 MAE: {metrics['MAE']:.2f}")
    print(f"📊 RMSE: {metrics['RMSE']:.2f}")
    print(f"📊 R²: {metrics['R2']:.4f}")
    
    # 4. GERAÇÃO DE PREVISÕES
    print("\n🎯 ETAPA 4: GERAÇÃO DE PREVISÕES")
    print("-" * 50)
    
    forecasts = generate_forecasts_2023(model, df_historical)
    print(f"✅ {len(forecasts):,} previsões geradas")
    
    # 5. ANÁLISE DOS RESULTADOS
    print("\n📊 ETAPA 5: ANÁLISE DOS RESULTADOS")
    print("-" * 50)
    
    analyze_forecasts(forecasts)
    
    # 6. SALVAMENTO
    output_file = save_results(forecasts)
    print(f"\n💾 Previsões salvas em: {output_file}")
    
    print("\n🎉 DEMO CONCLUÍDA COM SUCESSO!")
    print("=" * 60)
    return forecasts

def create_historical_data():
    """Criar dados históricos realistas para 2022"""
    data = []
    
    # PDVs e SKUs para simulação
    pdvs = [1023, 1045, 1067, 1089, 1012, 1156, 1198, 1234, 1345, 1456]
    skus = [f"SKU_{i:04d}" for i in range(1001, 1101)]  # 100 SKUs
    
    # Gerar dados para cada semana de 2022
    for semana in range(1, 53):  # 52 semanas
        for pdv in pdvs:
            # Cada PDV tem entre 15-25 SKUs diferentes por semana
            skus_ativos = np.random.choice(skus, size=np.random.randint(15, 26), replace=False)
            
            for sku in skus_ativos:
                # Fator sazonal (vendas maiores no fim do ano)
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (semana - 40) / 52)
                
                # Fator PDV (alguns PDVs vendem mais)
                pdv_factor = 1.0
                if pdv in [1023, 1045, 1198]:  # PDVs de alto volume
                    pdv_factor = 1.5
                elif pdv in [1067, 1234]:  # PDVs de baixo volume
                    pdv_factor = 0.7
                
                # Fator SKU (alguns produtos são mais populares)
                sku_num = int(sku.split('_')[1])
                if sku_num % 10 == 1:  # SKUs "premium"
                    sku_factor = 1.3
                elif sku_num % 10 == 9:  # SKUs "promocionais"
                    sku_factor = 1.8
                else:
                    sku_factor = 1.0
                
                # Quantidade base com variação aleatória
                base_qty = np.random.poisson(12)
                final_qty = max(1, int(base_qty * seasonal_factor * pdv_factor * sku_factor))
                
                data.append({
                    'ano': 2022,
                    'semana': semana,
                    'pdv': pdv,
                    'sku': sku,
                    'quantidade': final_qty
                })
    
    return pd.DataFrame(data)

def analyze_data(df):
    """Análise exploratória dos dados"""
    total_vendas = df['quantidade'].sum()
    media_vendas = df['quantidade'].mean()
    
    print(f"📊 Total de registros: {len(df):,}")
    print(f"🏪 PDVs únicos: {df['pdv'].nunique()}")
    print(f"📦 SKUs únicos: {df['sku'].nunique()}")
    print(f"📈 Total de vendas: {total_vendas:,} unidades")
    print(f"💡 Média por registro: {media_vendas:.1f} unidades")
    
    # Top performers
    print(f"\n🏆 TOP 5 PDVs por volume:")
    top_pdvs = df.groupby('pdv')['quantidade'].sum().nlargest(5)
    for i, (pdv, qty) in enumerate(top_pdvs.items(), 1):
        print(f"   {i}. PDV {pdv}: {qty:,} unidades")
    
    print(f"\n📦 TOP 5 SKUs por volume:")
    top_skus = df.groupby('sku')['quantidade'].sum().nlargest(5)
    for i, (sku, qty) in enumerate(top_skus.items(), 1):
        print(f"   {i}. {sku}: {qty:,} unidades")

def build_model(df):
    """Construir modelo de forecasting"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Preparar dados para o modelo
    df_model = df.copy()
    
    # Encoders para variáveis categóricas
    le_pdv = LabelEncoder()
    le_sku = LabelEncoder()
    
    df_model['pdv_encoded'] = le_pdv.fit_transform(df_model['pdv'])
    df_model['sku_encoded'] = le_sku.fit_transform(df_model['sku'])
    
    # Features adicionais
    df_model['trimestre'] = (df_model['semana'] - 1) // 13 + 1
    
    # Features e target
    feature_cols = ['semana', 'trimestre', 'pdv_encoded', 'sku_encoded']
    X = df_model[feature_cols]
    y = df_model['quantidade']
    
    # Divisão temporal (últimas 8 semanas para teste)
    train_mask = df_model['semana'] <= 44
    test_mask = df_model['semana'] > 44
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"📊 Dados de treino: {len(X_train):,} registros")
    print(f"📊 Dados de teste: {len(X_test):,} registros")
    
    # Treinar modelo
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Avaliar modelo
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }
    
    # Salvar encoders no modelo para uso posterior
    model.le_pdv = le_pdv
    model.le_sku = le_sku
    model.feature_cols = feature_cols
    
    return model, metrics

def generate_forecasts_2023(model, df_historical):
    """Gerar previsões para as 5 primeiras semanas de 2023"""
    forecasts = []
    
    # Obter PDVs e SKUs únicos do histórico
    pdvs_unicos = df_historical['pdv'].unique()
    skus_unicos = df_historical['sku'].unique()
    
    print(f"📊 Gerando previsões para {len(pdvs_unicos)} PDVs e {len(skus_unicos)} SKUs")
    
    for semana_2023 in range(1, 6):  # Semanas 1-5 de janeiro 2023
        print(f"   Processando semana {semana_2023}...")
        
        for pdv in pdvs_unicos:
            # Para cada PDV, usar apenas os SKUs que ele historicamente vende
            skus_pdv = df_historical[df_historical['pdv'] == pdv]['sku'].unique()
            
            # Limitar a 30 SKUs principais por PDV 
            if len(skus_pdv) > 30:
                top_skus = df_historical[
                    df_historical['pdv'] == pdv
                ].groupby('sku')['quantidade'].sum().nlargest(30).index
                skus_pdv = top_skus.values
            
            for sku in skus_pdv:
                try:
                    # Preparar features para predição
                    pdv_encoded = model.le_pdv.transform([pdv])[0]
                    sku_encoded = model.le_sku.transform([sku])[0]
                    trimestre = 1  # Janeiro = Q1
                    
                    X_pred = pd.DataFrame([{
                        'semana': semana_2023,
                        'trimestre': trimestre,
                        'pdv_encoded': pdv_encoded,
                        'sku_encoded': sku_encoded
                    }])
                    
                    # Fazer predição
                    pred_qty = model.predict(X_pred[model.feature_cols])[0]
                    pred_qty = max(1, int(round(pred_qty)))
                    
                    forecasts.append({
                        'semana': semana_2023,
                        'pdv': pdv,
                        'produto': sku,
                        'quantidade': pred_qty
                    })
                    
                except Exception:
                    # Se houver erro, usar média histórica
                    avg_qty = df_historical[
                        (df_historical['pdv'] == pdv) & 
                        (df_historical['sku'] == sku)
                    ]['quantidade'].mean()
                    
                    if pd.isna(avg_qty):
                        avg_qty = df_historical['quantidade'].mean()
                    
                    forecasts.append({
                        'semana': semana_2023,
                        'pdv': pdv,
                        'produto': sku,
                        'quantidade': max(1, int(round(avg_qty)))
                    })
    
    return pd.DataFrame(forecasts)

def analyze_forecasts(df_forecasts):
    """Analisar as previsões geradas"""
    total_previsoes = len(df_forecasts)
    total_quantidade = df_forecasts['quantidade'].sum()
    media_quantidade = df_forecasts['quantidade'].mean()
    
    print(f"📊 Total de previsões: {total_previsoes:,}")
    print(f"📈 Quantidade total prevista: {total_quantidade:,} unidades")
    print(f"💡 Média por previsão: {media_quantidade:.1f} unidades")
    
    # Análise por semana
    print(f"\n📅 Previsões por semana (Janeiro 2023):")
    por_semana = df_forecasts.groupby('semana')['quantidade'].sum()
    for semana in por_semana.index:
        total = por_semana.loc[semana]
        print(f"   Semana {semana}: {total:,} unidades")
    
    # Top PDVs previstos
    print(f"\n🏪 TOP 5 PDVs - Volume Previsto:")
    top_pdvs_pred = df_forecasts.groupby('pdv')['quantidade'].sum().nlargest(5)
    for i, (pdv, qty) in enumerate(top_pdvs_pred.items(), 1):
        print(f"   {i}. PDV {pdv}: {qty:,} unidades")

def save_results(df_forecasts):
    """Salvar resultados em CSV"""
    import os
    os.makedirs('data/processed', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/processed/forecast_janeiro_2023_{timestamp}.csv"
    
    df_forecasts.to_csv(filename, index=False)
    return filename

if __name__ == "__main__":
    forecasts = main()
