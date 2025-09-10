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
    print(f"📊 Período: {df_historical.shape[0]} registros semanais")
    
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
                # Simulação de quantidade com padrões realistas
                
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

def analyze_data(df):\n    """Análise exploratória dos dados"""\n    total_vendas = df['quantidade'].sum()\n    media_vendas = df['quantidade'].mean()\n    \n    print(f\"📊 Total de registros: {len(df):,}\")\n    print(f\"🏪 PDVs únicos: {df['pdv'].nunique()}\")\n    print(f\"📦 SKUs únicos: {df['sku'].nunique()}\")\n    print(f\"📈 Total de vendas: {total_vendas:,} unidades\")\n    print(f\"💡 Média por registro: {media_vendas:.1f} unidades\")\n    \n    # Top performers\n    print(f\"\\n🏆 TOP 5 PDVs por volume:\")\n    top_pdvs = df.groupby('pdv')['quantidade'].sum().nlargest(5)\n    for i, (pdv, qty) in enumerate(top_pdvs.items(), 1):\n        print(f\"   {i}. PDV {pdv}: {qty:,} unidades\")\n    \n    print(f\"\\n📦 TOP 5 SKUs por volume:\")\n    top_skus = df.groupby('sku')['quantidade'].sum().nlargest(5)\n    for i, (sku, qty) in enumerate(top_skus.items(), 1):\n        print(f\"   {i}. {sku}: {qty:,} unidades\")\n    \n    # Análise sazonal\n    vendas_por_trimestre = df.groupby(df['semana'] // 13 + 1)['quantidade'].sum()\n    print(f\"\\n📅 Vendas por trimestre:\")\n    for trimestre, vendas in vendas_por_trimestre.items():\n        print(f\"   Q{trimestre}: {vendas:,} unidades\")\n\ndef build_model(df):\n    """Construir modelo de forecasting\"\"\"    \n    from sklearn.ensemble import RandomForestRegressor\n    from sklearn.preprocessing import LabelEncoder\n    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n    \n    # Preparar dados para o modelo\n    df_model = df.copy()\n    \n    # Encoders para variáveis categóricas\n    le_pdv = LabelEncoder()\n    le_sku = LabelEncoder()\n    \n    df_model['pdv_encoded'] = le_pdv.fit_transform(df_model['pdv'])\n    df_model['sku_encoded'] = le_sku.fit_transform(df_model['sku'])\n    \n    # Features adicionais\n    df_model['trimestre'] = (df_model['semana'] - 1) // 13 + 1\n    df_model['semana_trimestre'] = df_model['semana'] % 13\n    \n    # Features e target\n    feature_cols = ['semana', 'trimestre', 'semana_trimestre', 'pdv_encoded', 'sku_encoded']\n    X = df_model[feature_cols]\n    y = df_model['quantidade']\n    \n    # Divisão temporal (últimas 8 semanas para teste)\n    train_mask = df_model['semana'] <= 44\n    test_mask = df_model['semana'] > 44\n    \n    X_train = X[train_mask]\n    y_train = y[train_mask]\n    X_test = X[test_mask]\n    y_test = y[test_mask]\n    \n    print(f\"📊 Dados de treino: {len(X_train):,} registros\")\n    print(f\"📊 Dados de teste: {len(X_test):,} registros\")\n    \n    # Treinar modelo\n    model = RandomForestRegressor(\n        n_estimators=100,\n        max_depth=15,\n        min_samples_split=5,\n        random_state=42,\n        n_jobs=-1\n    )\n    \n    model.fit(X_train, y_train)\n    \n    # Avaliar modelo\n    y_pred = model.predict(X_test)\n    \n    mae = mean_absolute_error(y_test, y_pred)\n    mse = mean_squared_error(y_test, y_pred)\n    rmse = np.sqrt(mse)\n    r2 = r2_score(y_test, y_pred)\n    \n    metrics = {\n        'MAE': mae,\n        'MSE': mse,\n        'RMSE': rmse,\n        'R2': r2\n    }\n    \n    # Salvar encoders no modelo para uso posterior\n    model.le_pdv = le_pdv\n    model.le_sku = le_sku\n    model.feature_cols = feature_cols\n    \n    return model, metrics\n\ndef generate_forecasts_2023(model, df_historical):\n    """Gerar previsões para as 5 primeiras semanas de 2023\"\"\"\n    forecasts = []\n    \n    # Obter PDVs e SKUs únicos do histórico\n    pdvs_unicos = df_historical['pdv'].unique()\n    skus_unicos = df_historical['sku'].unique()\n    \n    print(f\"📊 Gerando previsões para {len(pdvs_unicos)} PDVs e {len(skus_unicos)} SKUs\")\n    \n    for semana_2023 in range(1, 6):  # Semanas 1-5 de janeiro 2023\n        print(f\"   Processando semana {semana_2023}...\")\n        \n        for pdv in pdvs_unicos:\n            # Para cada PDV, usar apenas os SKUs que ele historicamente vende\n            skus_pdv = df_historical[df_historical['pdv'] == pdv]['sku'].unique()\n            \n            # Limitar a 30 SKUs principais por PDV para não ficar muito grande\n            if len(skus_pdv) > 30:\n                # Selecionar os 30 SKUs com maior volume histórico\n                top_skus = df_historical[\n                    df_historical['pdv'] == pdv\n                ].groupby('sku')['quantidade'].sum().nlargest(30).index\n                skus_pdv = top_skus.values\n            \n            for sku in skus_pdv:\n                try:\n                    # Preparar features para predição\n                    pdv_encoded = model.le_pdv.transform([pdv])[0]\n                    sku_encoded = model.le_sku.transform([sku])[0]\n                    \n                    trimestre = 1  # Janeiro = Q1\n                    semana_trimestre = semana_2023\n                    \n                    X_pred = pd.DataFrame([{\n                        'semana': semana_2023,\n                        'trimestre': trimestre,\n                        'semana_trimestre': semana_trimestre,\n                        'pdv_encoded': pdv_encoded,\n                        'sku_encoded': sku_encoded\n                    }])\n                    \n                    # Fazer predição\n                    pred_qty = model.predict(X_pred[model.feature_cols])[0]\n                    pred_qty = max(1, int(round(pred_qty)))\n                    \n                    forecasts.append({\n                        'semana': semana_2023,\n                        'pdv': pdv,\n                        'produto': sku,\n                        'quantidade': pred_qty\n                    })\n                    \n                except Exception:\n                    # Se houver erro (ex: SKU/PDV novo), usar média histórica\n                    avg_qty = df_historical[\n                        (df_historical['pdv'] == pdv) & \n                        (df_historical['sku'] == sku)\n                    ]['quantidade'].mean()\n                    \n                    if pd.isna(avg_qty):\n                        avg_qty = df_historical['quantidade'].mean()\n                    \n                    forecasts.append({\n                        'semana': semana_2023,\n                        'pdv': pdv,\n                        'produto': sku,\n                        'quantidade': max(1, int(round(avg_qty)))\n                    })\n    \n    return pd.DataFrame(forecasts)\n\ndef analyze_forecasts(df_forecasts):\n    \"\"\"Analisar as previsões geradas\"\"\"\n    total_previsoes = len(df_forecasts)\n    total_quantidade = df_forecasts['quantidade'].sum()\n    media_quantidade = df_forecasts['quantidade'].mean()\n    \n    print(f\"📊 Total de previsões: {total_previsoes:,}\")\n    print(f\"📈 Quantidade total prevista: {total_quantidade:,} unidades\")\n    print(f\"💡 Média por previsão: {media_quantidade:.1f} unidades\")\n    \n    # Análise por semana\n    print(f\"\\n📅 Previsões por semana (Janeiro 2023):\")\n    por_semana = df_forecasts.groupby('semana')['quantidade'].agg(['sum', 'count', 'mean'])\n    for semana in por_semana.index:\n        total = por_semana.loc[semana, 'sum']\n        count = por_semana.loc[semana, 'count']\n        avg = por_semana.loc[semana, 'mean']\n        print(f\"   Semana {semana}: {total:,} unidades ({count:,} previsões, média: {avg:.1f})\")\n    \n    # Top PDVs previstos\n    print(f\"\\n🏪 TOP 5 PDVs - Volume Previsto:\")\n    top_pdvs_pred = df_forecasts.groupby('pdv')['quantidade'].sum().nlargest(5)\n    for i, (pdv, qty) in enumerate(top_pdvs_pred.items(), 1):\n        print(f\"   {i}. PDV {pdv}: {qty:,} unidades\")\n    \n    # Top SKUs previstos\n    print(f\"\\n📦 TOP 5 Produtos - Volume Previsto:\")\n    top_skus_pred = df_forecasts.groupby('produto')['quantidade'].sum().nlargest(5)\n    for i, (sku, qty) in enumerate(top_skus_pred.items(), 1):\n        print(f\"   {i}. {sku}: {qty:,} unidades\")\n\ndef save_results(df_forecasts):\n    \"\"\"Salvar resultados em CSV\"\"\"\n    timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n    filename = f\"forecast_janeiro_2023_{timestamp}.csv\"\n    \n    df_forecasts.to_csv(filename, index=False)\n    return filename\n\nif __name__ == \"__main__\":\n    forecasts = main()
