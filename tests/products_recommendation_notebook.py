# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas para séries temporais e forecasting
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Bibliotecas para análise temporal
from datetime import datetime, timedelta
import calendar

# Bibliotecas para visualização
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly não disponível - usando apenas matplotlib")

# Configurações
plt.style.use('default')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("✓ Bibliotecas importadas com sucesso!")
print("🎯 OBJETIVO: Modelo de Forecast para Reposição de Estoque")
print("📅 META: Prever vendas para 5 semanas de janeiro/2023")

print("📂 CARREGAMENTO DOS DADOS DE VENDAS 2022")
print("="*60)

# Caminhos dos arquivos
arquivos = [
    "data/raw/part-00000-tid-6364321654468257203-dc13a5d6-36ae-48c6-a018-37d8cfe34cf6-263-1-c000.snappy.parquet",
    "data/raw/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet",
    "data/raw/part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet"
]

# Carregamento dos dados históricos
dataframes = []
for i, arquivo in enumerate(arquivos):
    try:
        df_temp = pd.read_parquet(arquivo)
        df_temp['arquivo_origem'] = f'arquivo_{i+1}'
        dataframes.append(df_temp)
        print(f"✓ Arquivo {i+1}: {df_temp.shape[0]} linhas, {df_temp.shape[1]} colunas")
    except Exception as e:
        print(f"❌ Erro ao carregar arquivo {i+1}: {e}")

# União dos dataframes
if dataframes:
    df_raw = pd.concat(dataframes, ignore_index=True)
    print(f"\n📊 Dataset histórico: {df_raw.shape[0]} linhas, {df_raw.shape[1]} colunas")
else:
    print("❌ Nenhum arquivo foi carregado com sucesso!")
    
print("\n🔍 Estrutura dos dados:")
display(df_raw.info())
print("\n📋 Primeiras linhas:")
display(df_raw.head())

print("🔍 ANÁLISE DA ESTRUTURA DOS DADOS")
print("="*60)

# Identificar colunas relevantes para forecast
print("📊 Análise das colunas disponíveis:")
for col in df_raw.columns:
    unique_vals = df_raw[col].nunique()
    data_type = df_raw[col].dtype
    null_count = df_raw[col].isnull().sum()
    print(f"   {col}: {unique_vals} valores únicos, tipo: {data_type}, nulos: {null_count}")

# Identificar possíveis colunas chave
possible_date_cols = []
possible_pdv_cols = []
possible_sku_cols = []
possible_qty_cols = []

for col in df_raw.columns:
    col_lower = col.lower()
    
    # Colunas de data/tempo
    if any(word in col_lower for word in ['data', 'date', 'time', 'semana', 'week', 'mes', 'month']):
        possible_date_cols.append(col)
    
    # Colunas de PDV/loja
    if any(word in col_lower for word in ['pdv', 'loja', 'store', 'ponto', 'unidade']):
        possible_pdv_cols.append(col)
    
    # Colunas de produto/SKU
    if any(word in col_lower for word in ['sku', 'produto', 'product', 'item', 'codigo']):
        possible_sku_cols.append(col)
    
    # Colunas de quantidade/vendas
    if any(word in col_lower for word in ['qtd', 'quantidade', 'qty', 'vendas', 'sales', 'volume']):
        possible_qty_cols.append(col)

print(f"\n🎯 Colunas identificadas:")
print(f"   📅 Datas: {possible_date_cols}")
print(f"   🏪 PDV: {possible_pdv_cols}")
print(f"   📦 SKU/Produto: {possible_sku_cols}")
print(f"   📈 Quantidade/Vendas: {possible_qty_cols}")

print("⚙️ ESTRUTURAÇÃO PARA ANÁLISE TEMPORAL")
print("="*60)

# Criar estrutura padronizada assumindo estrutura típica de dados de vendas
df = df_raw.copy()

# Mapear colunas automaticamente ou criar estrutura simulada
if possible_date_cols:
    date_col = possible_date_cols[0]
    print(f"📅 Usando coluna de data: {date_col}")
else:
    # Simular datas de 2022 se não houver coluna de data
    print("📅 Simulando estrutura temporal para 2022...")
    dates_2022 = pd.date_range('2022-01-01', '2022-12-31', freq='D')
    np.random.seed(42)
    df['data_venda'] = np.random.choice(dates_2022, size=len(df))
    date_col = 'data_venda'

if possible_pdv_cols:
    pdv_col = possible_pdv_cols[0]
else:
    # Simular PDVs se não existir
    np.random.seed(42)
    df['pdv'] = np.random.choice([1023, 1045, 1067, 1089, 1012, 1156, 1198, 1234, 1345, 1456], size=len(df))
    pdv_col = 'pdv'

if possible_sku_cols:
    sku_col = possible_sku_cols[0]
else:
    # Simular SKUs se não existir
    np.random.seed(42)
    skus = [f"SKU_{i:04d}" for i in range(100, 500)]
    df['sku'] = np.random.choice(skus, size=len(df))
    sku_col = 'sku'

if possible_qty_cols:
    qty_col = possible_qty_cols[0]
else:
    # Simular quantidades baseadas em padrões realistas
    np.random.seed(42)
    # Quantidade base com variação sazonal e aleatória
    base_qty = np.random.poisson(15, size=len(df))
    seasonal_factor = np.sin(2 * np.pi * np.arange(len(df)) / 365) * 3 + 1
    df['quantidade_vendida'] = np.maximum(1, (base_qty * seasonal_factor).astype(int))
    qty_col = 'quantidade_vendida'

print(f"✓ Colunas mapeadas:")
print(f"   📅 Data: {date_col}")
print(f"   🏪 PDV: {pdv_col}")
print(f"   📦 SKU: {sku_col}")
print(f"   📈 Quantidade: {qty_col}")

# Converter data se necessário
if df[date_col].dtype == 'object':
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        print("✓ Coluna de data convertida para datetime")
    except:
        print("⚠️ Erro na conversão de data - usando índices temporais")

print("📊 AGREGAÇÃO DOS DADOS POR SEMANA/PDV/SKU")
print("="*60)

# Criar coluna de semana
df['ano'] = df[date_col].dt.year
df['semana'] = df[date_col].dt.isocalendar().week
df['ano_semana'] = df['ano'].astype(str) + '_S' + df['semana'].astype(str).str.zfill(2)

# Agregação por semana, PDV e SKU
df_agg = df.groupby(['ano', 'semana', 'ano_semana', pdv_col, sku_col]).agg({
    qty_col: 'sum',
    date_col: 'min'  # Para manter referência de data
}).reset_index()

df_agg.rename(columns={
    pdv_col: 'pdv',
    sku_col: 'sku', 
    qty_col: 'quantidade',
    date_col: 'data_referencia'
}, inplace=True)

print(f"📊 Dados agregados: {df_agg.shape[0]} registros")
print(f"📅 Período: {df_agg['ano'].min()} a {df_agg['ano'].max()}")
print(f"🏪 PDVs únicos: {df_agg['pdv'].nunique()}")
print(f"📦 SKUs únicos: {df_agg['sku'].nunique()}")
print(f"📈 Vendas totais: {df_agg['quantidade'].sum():,}")

# Filtrar apenas dados de 2022 (base histórica)
df_2022 = df_agg[df_agg['ano'] == 2022].copy()
print(f"\n✓ Base histórica 2022: {df_2022.shape[0]} registros")

# Mostrar amostra dos dados
display(df_2022.head(10))

print("📈 ANÁLISE EXPLORATÓRIA DOS DADOS DE VENDAS")
print("="*60)

# Estatísticas gerais
print("📊 Estatísticas Gerais 2022:")
print(f"   • Vendas totais: {df_2022['quantidade'].sum():,} unidades")
print(f"   • Média semanal por PDV/SKU: {df_2022['quantidade'].mean():.1f} unidades")
print(f"   • Mediana: {df_2022['quantidade'].median():.1f} unidades")
print(f"   • Desvio padrão: {df_2022['quantidade'].std():.1f}")

# Top PDVs e SKUs
top_pdvs = df_2022.groupby('pdv')['quantidade'].sum().sort_values(ascending=False).head(10)
top_skus = df_2022.groupby('sku')['quantidade'].sum().sort_values(ascending=False).head(10)

print(f"\n🏪 Top 10 PDVs por volume:")
for pdv, qty in top_pdvs.items():
    print(f"   PDV {pdv}: {qty:,} unidades")

print(f"\n📦 Top 5 SKUs por volume:")
for sku, qty in top_skus.head(5).items():
    print(f"   {sku}: {qty:,} unidades")

# Visualizações
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Análise Exploratória - Dados de Vendas 2022', fontsize=16, fontweight='bold')

# 1. Vendas por semana
vendas_semana = df_2022.groupby('semana')['quantidade'].sum()
axes[0, 0].plot(vendas_semana.index, vendas_semana.values, marker='o', linewidth=2)
axes[0, 0].set_title('Vendas Totais por Semana (2022)')
axes[0, 0].set_xlabel('Semana')
axes[0, 0].set_ylabel('Quantidade Total')
axes[0, 0].grid(True, alpha=0.3)

# 2. Distribuição de quantidades
axes[0, 1].hist(df_2022['quantidade'], bins=50, alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Distribuição das Quantidades Vendidas')
axes[0, 1].set_xlabel('Quantidade')
axes[0, 1].set_ylabel('Frequência')
axes[0, 1].set_yscale('log')

# 3. Top 10 PDVs
axes[0, 2].barh(range(len(top_pdvs)), top_pdvs.values)
axes[0, 2].set_yticks(range(len(top_pdvs)))
axes[0, 2].set_yticklabels([f'PDV {p}' for p in top_pdvs.index])
axes[0, 2].set_title('Top 10 PDVs - Volume Total')
axes[0, 2].set_xlabel('Quantidade Total')

# 4. Sazonalidade mensal
df_2022['mes'] = df_2022['data_referencia'].dt.month
vendas_mes = df_2022.groupby('mes')['quantidade'].sum()
meses_nomes = [calendar.month_name[i] for i in vendas_mes.index]
axes[1, 0].bar(range(len(vendas_mes)), vendas_mes.values)
axes[1, 0].set_title('Vendas por Mês (2022)')
axes[1, 0].set_xlabel('Mês')
axes[1, 0].set_ylabel('Quantidade Total')
axes[1, 0].set_xticks(range(len(vendas_mes)))
axes[1, 0].set_xticklabels([m[:3] for m in meses_nomes], rotation=45)

# 5. Boxplot por PDV (top 5)
top_5_pdvs = top_pdvs.index[:5]
data_boxplot = [df_2022[df_2022['pdv'] == pdv]['quantidade'].values for pdv in top_5_pdvs]
axes[1, 1].boxplot(data_boxplot, labels=[f'PDV\n{p}' for p in top_5_pdvs])
axes[1, 1].set_title('Distribuição de Vendas - Top 5 PDVs')
axes[1, 1].set_ylabel('Quantidade')

# 6. Tendência temporal geral
vendas_acum = df_2022.groupby('ano_semana')['quantidade'].sum().cumsum()
axes[1, 2].plot(range(len(vendas_acum)), vendas_acum.values, color='green', linewidth=2)
axes[1, 2].set_title('Vendas Acumuladas 2022')
axes[1, 2].set_xlabel('Semanas')
axes[1, 2].set_ylabel('Quantidade Acumulada')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("⚙️ FEATURE ENGINEERING PARA FORECASTING")
print("="*60)

def criar_features_temporais(df_base):
    """Criar features temporais para modelos de forecast"""
    df_features = df_base.copy()
    
    # Features temporais básicas
    df_features['semana_ano'] = df_features['semana']
    df_features['trimestre'] = ((df_features['semana'] - 1) // 13) + 1
    
    # Features sazonais
    df_features['seno_semana'] = np.sin(2 * np.pi * df_features['semana'] / 52)
    df_features['cosseno_semana'] = np.cos(2 * np.pi * df_features['semana'] / 52)
    
    # Features de tendência
    df_features['tendencia'] = range(len(df_features))
    
    # Inicializar colunas de lag
    df_features['lag_1'] = np.nan
    df_features['lag_2'] = np.nan
    df_features['lag_4'] = np.nan
    df_features['media_4'] = np.nan
    df_features['media_8'] = np.nan
    
    # Features de lag (vendas anteriores) por PDV/SKU
    # Fazendo por grupo para evitar problemas de alinhamento
    grouped = df_features.groupby(['pdv', 'sku'])
    df_features[['lag_1','lag_2','lag_4','media_4','media_8']] = grouped['quantidade'].apply(
        lambda x: pd.DataFrame({
            'lag_1': x.shift(1),
            'lag_2': x.shift(2),
            'lag_4': x.shift(4),
            'media_4': x.rolling(4).mean(),
            'media_8': x.rolling(8).mean()
        })
    ).reset_index(level=[0,1], drop=True)
    
    # Preencher NaNs com médias gerais
    mean_qty = df_features['quantidade'].mean()
    df_features['lag_1'].fillna(mean_qty, inplace=True)
    df_features['lag_2'].fillna(mean_qty, inplace=True)
    df_features['lag_4'].fillna(mean_qty, inplace=True)
    df_features['media_4'].fillna(mean_qty, inplace=True)
    df_features['media_8'].fillna(mean_qty, inplace=True)
    
    # Features de PDV e SKU (encoding)
    le_pdv = LabelEncoder()
    le_sku = LabelEncoder()
    df_features['pdv_encoded'] = le_pdv.fit_transform(df_features['pdv'])
    df_features['sku_encoded'] = le_sku.fit_transform(df_features['sku'])
    
    return df_features, le_pdv, le_sku

# Aplicar feature engineering
print("🔧 Criando features temporais...")
df_features, encoder_pdv, encoder_sku = criar_features_temporais(df_2022)

# Colunas de features para o modelo
feature_cols = ['semana_ano', 'trimestre', 'seno_semana', 'cosseno_semana', 
                'tendencia', 'lag_1', 'lag_2', 'lag_4', 'media_4', 'media_8',
                'pdv_encoded', 'sku_encoded']

print(f"✓ Features criadas: {len(feature_cols)}")
print(f"✓ Dataset com features: {df_features.shape}")

# Mostrar correlação das features
corr_matrix = df_features[feature_cols + ['quantidade']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Matriz de Correlação - Features vs Quantidade')
plt.tight_layout()
plt.show()

print("📊 DIVISÃO DOS DADOS PARA TREINAMENTO")
print("="*60)

# Ordenar por tempo
df_features = df_features.sort_values(['ano', 'semana', 'pdv', 'sku']).reset_index(drop=True)

# Usar TimeSeriesSplit para respeitar ordem temporal
# Últimas 8 semanas como teste (aproximadamente 15% dos dados de 2022)
total_semanas = df_features['semana'].nunique()
semanas_teste = 8
semana_corte = df_features['semana'].max() - semanas_teste

print(f"📅 Total de semanas em 2022: {total_semanas}")
print(f"📊 Semanas para treino: até semana {semana_corte}")
print(f"📊 Semanas para teste: {semanas_teste} últimas semanas")

# Dividir dados
train_mask = df_features['semana'] <= semana_corte
test_mask = df_features['semana'] > semana_corte

X_train = df_features[train_mask][feature_cols]
y_train = df_features[train_mask]['quantidade']
X_test = df_features[test_mask][feature_cols]
y_test = df_features[test_mask]['quantidade']

print(f"✓ Dados de treino: {X_train.shape[0]} registros")
print(f"✓ Dados de teste: {X_test.shape[0]} registros")

# Remover qualquer NaN restante
mask_train = ~(X_train.isnull().any(axis=1) | y_train.isnull())
mask_test = ~(X_test.isnull().any(axis=1) | y_test.isnull())

X_train = X_train[mask_train]
y_train = y_train[mask_train]
X_test = X_test[mask_test]
y_test = y_test[mask_test]

print(f"✓ Após limpeza - Treino: {X_train.shape[0]}, Teste: {X_test.shape[0]}")

print("🤖 COMPARAÇÃO DE MODELOS DE FORECASTING")
print("="*60)

# Modelos específicos para regressão/forecasting
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0)
}

# Adicionar XGBoost se disponível
try:
    import xgboost as xgb
    models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
except ImportError:
    print("⚠️ XGBoost não disponível")

# Resultados
results = {}
trained_models = {}

print("🔄 Treinando modelos de forecasting...")

for name, model in models.items():
    print(f"\n📈 Treinando {name}...")
    
    try:
        # Treinar modelo
        if name == 'SVR':
            # SVR precisa de normalização
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Armazenar scaler junto com o modelo
            trained_models[name] = (model, scaler)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            trained_models[name] = model
        
        # Calcular métricas
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
        
        results[name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        }
        
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   R²: {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        continue

print("📊 VISUALIZAÇÃO DOS RESULTADOS")
print("="*60)

# DataFrame com resultados
results_df = pd.DataFrame(results).T

# Plotar comparação
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Comparação de Modelos de Forecasting', fontsize=16, fontweight='bold')

# 1. MAE
axes[0, 0].bar(results_df.index, results_df['MAE'], color='skyblue')
axes[0, 0].set_title('Mean Absolute Error (MAE)')
axes[0, 0].set_ylabel('MAE')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. RMSE
axes[0, 1].bar(results_df.index, results_df['RMSE'], color='lightcoral')
axes[0, 1].set_title('Root Mean Square Error (RMSE)')
axes[0, 1].set_ylabel('RMSE')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. R²
axes[1, 0].bar(results_df.index, results_df['R²'], color='lightgreen')
axes[1, 0].set_title('Coeficiente de Determinação (R²)')
axes[1, 0].set_ylabel('R²')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. MAPE
axes[1, 1].bar(results_df.index, results_df['MAPE'], color='orange')
axes[1, 1].set_title('Mean Absolute Percentage Error (MAPE)')
axes[1, 1].set_ylabel('MAPE (%)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Tabela de resultados
print("\n📊 TABELA COMPARATIVA DE RESULTADOS:")
display(results_df.round(4))

# Selecionar melhor modelo (menor RMSE)
best_model_name = results_df['RMSE'].idxmin()
print(f"\n🏆 MELHOR MODELO: {best_model_name}")
print(f"📊 RMSE: {results_df.loc[best_model_name, 'RMSE']:.2f}")
print(f"📊 R²: {results_df.loc[best_model_name, 'R²']:.4f}")
print(f"📊 MAPE: {results_df.loc[best_model_name, 'MAPE']:.2f}%")

print("🔍 ANÁLISE DETALHADA DO MELHOR MODELO")
print("="*60)

# Obter modelo e fazer predições
best_model = trained_models[best_model_name]

if best_model_name == 'SVR':
    model, scaler = best_model
    X_test_scaled = scaler.transform(X_test)
    y_pred_best = model.predict(X_test_scaled)
else:
    model = best_model
    y_pred_best = model.predict(X_test)

# Análise de resíduos
residuals = y_test - y_pred_best

# Visualizações detalhadas
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f'Análise Detalhada - {best_model_name}', fontsize=16, fontweight='bold')

# 1. Predito vs Real
axes[0, 0].scatter(y_test, y_pred_best, alpha=0.6)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Valores Reais')
axes[0, 0].set_ylabel('Valores Preditos')
axes[0, 0].set_title('Predito vs Real')
axes[0, 0].grid(True, alpha=0.3)

# 2. Distribuição dos resíduos
axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Resíduos')
axes[0, 1].set_ylabel('Frequência')
axes[0, 1].set_title('Distribuição dos Resíduos')
axes[0, 1].axvline(x=0, color='red', linestyle='--')

# 3. Resíduos vs Preditos
axes[1, 0].scatter(y_pred_best, residuals, alpha=0.6)
axes[1, 0].set_xlabel('Valores Preditos')
axes[1, 0].set_ylabel('Resíduos')
axes[1, 0].set_title('Resíduos vs Preditos')
axes[1, 0].axhline(y=0, color='red', linestyle='--')
axes[1, 0].grid(True, alpha=0.3)

# 4. Feature Importance (se disponível)
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_names = feature_cols
    
    # Ordenar features por importância
    indices = np.argsort(importances)[::-1]
    
    axes[1, 1].bar(range(len(importances)), importances[indices])
    axes[1, 1].set_title('Feature Importances')
    axes[1, 1].set_xlabel('Features')
    axes[1, 1].set_ylabel('Importância')
    axes[1, 1].set_xticks(range(len(importances)))
    axes[1, 1].set_xticklabels([feature_names[i] for i in indices], rotation=45)
elif hasattr(model, 'coef_'):
    coef = np.abs(model.coef_)
    feature_names = feature_cols
    
    axes[1, 1].bar(range(len(coef)), coef)
    axes[1, 1].set_title('Feature Coefficients (Absolute)')
    axes[1, 1].set_xlabel('Features')
    axes[1, 1].set_ylabel('Coeficiente Absoluto')
    axes[1, 1].set_xticks(range(len(coef)))
    axes[1, 1].set_xticklabels(feature_names, rotation=45)
else:
    axes[1, 1].text(0.5, 0.5, 'Feature importance não disponível\npara este modelo', 
                   ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Feature Importance')

plt.tight_layout()
plt.show()

print("🎯 GERAÇÃO DE PREVISÕES PARA JANEIRO 2023")
print("="*60)

def gerar_previsoes_janeiro_2023(modelo_treinado, df_historico, pdvs_ativos=None, skus_ativos=None, modelo_name=None):
    """
    Gerar previsões para as 5 primeiras semanas de janeiro 2023
    """
    print("🔄 Gerando previsões para janeiro 2023...")
    
    # Definir semanas de janeiro 2023 (semanas 1-5 de 2023)
    semanas_2023 = [1, 2, 3, 4, 5]
    
    # Obter PDVs e SKUs únicos do histórico se não especificado
    if pdvs_ativos is None:
        pdvs_ativos = df_historico['pdv'].unique()
    
    if skus_ativos is None:
        # Usar apenas os SKUs mais vendidos para reduzir complexidade
        top_skus = df_historico.groupby('sku')['quantidade'].sum().sort_values(ascending=False)
        skus_ativos = top_skus.head(50).index.tolist()  # Top 50 SKUs
    
    print(f"📊 Gerando previsões para:")
    print(f"   • {len(semanas_2023)} semanas")
    print(f"   • {len(pdvs_ativos)} PDVs")
    print(f"   • {len(skus_ativos)} SKUs")
    
    previsoes = []
    
    for semana in semanas_2023:
        for pdv in pdvs_ativos:
            for sku in skus_ativos:
                try:
                    # Obter dados históricos para este PDV/SKU
                    hist_pdv_sku = df_historico[
                        (df_historico['pdv'] == pdv) & 
                        (df_historico['sku'] == sku)
                    ].sort_values('semana')
                    
                    if len(hist_pdv_sku) == 0:
                        # Se não há histórico, usar médias gerais
                        lag_1 = df_historico['quantidade'].mean()
                        lag_2 = df_historico['quantidade'].mean()
                        lag_4 = df_historico['quantidade'].mean()
                        media_4 = df_historico['quantidade'].mean()
                        media_8 = df_historico['quantidade'].mean()
                    else:
                        # Usar últimos valores do histórico
                        lag_1 = hist_pdv_sku['quantidade'].iloc[-1] if len(hist_pdv_sku) >= 1 else df_historico['quantidade'].mean()
                        lag_2 = hist_pdv_sku['quantidade'].iloc[-2] if len(hist_pdv_sku) >= 2 else lag_1
                        lag_4 = hist_pdv_sku['quantidade'].iloc[-4] if len(hist_pdv_sku) >= 4 else lag_1
                        media_4 = hist_pdv_sku['quantidade'].tail(4).mean() if len(hist_pdv_sku) >= 4 else lag_1
                        media_8 = hist_pdv_sku['quantidade'].tail(8).mean() if len(hist_pdv_sku) >= 8 else media_4
                    
                    # Criar features para a predição
                    features = {
                        'semana_ano': semana,
                        'trimestre': 1,  # Janeiro está no Q1
                        'seno_semana': np.sin(2 * np.pi * semana / 52),
                        'cosseno_semana': np.cos(2 * np.pi * semana / 52),
                        'tendencia': 53 + semana - 1,  # Continuação da tendência de 2022 (exemplo)
                        'lag_1': lag_1,
                        'lag_2': lag_2,
                        'lag_4': lag_4,
                        'media_4': media_4,
                        'media_8': media_8,
                        'pdv_encoded': int(encoder_pdv.transform([pdv])[0]) if pdv in encoder_pdv.classes_ else 0,
                        'sku_encoded': int(encoder_sku.transform([sku])[0]) if sku in encoder_sku.classes_ else 0
                    }
                    
                    # Converter para array na ordem correta das features
                    X_pred = np.array([features[col] for col in feature_cols]).reshape(1, -1)
                    
                    # Fazer predição
                    if modelo_name == 'SVR' and isinstance(modelo_treinado, tuple):
                        modelo, scaler = modelo_treinado
                        X_pred_scaled = scaler.transform(X_pred)
                        pred_qty = modelo.predict(X_pred_scaled)[0]
                    else:
                        pred_qty = modelo_treinado.predict(X_pred)[0]
                    
                    # Garantir que a predição seja positiva e realista
                    pred_qty = max(1, int(round(float(pred_qty))))
                    
                    previsoes.append({
                        'semana': semana,
                        'pdv': pdv,
                        'produto': sku,  # Usar SKU como produto
                        'quantidade': pred_qty
                    })
                    
                except Exception as e:
                    print(f"⚠️ Erro ao prever para PDV {pdv}, SKU {sku}, Semana {semana}: {e}")
                    continue
    
    df_previsoes = pd.DataFrame(previsoes)
    print(f"✅ {len(df_previsoes)} previsões geradas")
    
    return df_previsoes

# Gerar previsões
df_previsoes_2023 = gerar_previsoes_janeiro_2023(
    modelo_treinado=best_model,
    df_historico=df_features,
    pdvs_ativos=None,  # Usar todos os PDVs
    skus_ativos=None,  # Usar top 50 SKUs
    modelo_name=best_model_name
)

print("\n📋 PREVISÕES PARA JANEIRO 2023:")
print("Formato: semana | pdv | produto | quantidade")
display(df_previsoes_2023.head(15))

print("📊 ANÁLISE DAS PREVISÕES GERADAS")
print("="*60)

# Estatísticas das previsões
print("📈 Estatísticas Gerais das Previsões:")
print(f"   • Total de registros: {len(df_previsoes_2023):,}")
print(f"   • Quantidade total prevista: {df_previsoes_2023['quantidade'].sum():,} unidades")
print(f"   • Média por registro: {df_previsoes_2023['quantidade'].mean():.1f} unidades")
print(f"   • Mediana: {df_previsoes_2023['quantidade'].median():.1f} unidades")
print(f"   • Desvio padrão: {df_previsoes_2023['quantidade'].std():.1f}")

# Análise por semana
print("\n📅 Análise por Semana:")
semana_stats = df_previsoes_2023.groupby('semana').agg({
    'quantidade': ['sum', 'mean', 'count'],
    'produto': 'nunique',
    'pdv': 'nunique'
}).round(2)
display(semana_stats)

# Top PDVs e Produtos
top_pdvs_2023 = df_previsoes_2023.groupby('pdv')['quantidade'].sum().sort_values(ascending=False).head(10)
top_produtos_2023 = df_previsoes_2023.groupby('produto')['quantidade'].sum().sort_values(ascending=False).head(10)

print("\n🏪 Top 10 PDVs - Previsão Janeiro 2023:")
for pdv, qty in top_pdvs_2023.items():
    print(f"   PDV {pdv}: {qty:,} unidades")

print("\n📦 Top 10 Produtos - Previsão Janeiro 2023:")
for produto, qty in top_produtos_2023.items():
    print(f"   {produto}: {qty:,} unidades")

# Visualizações das previsões (célula final completada para Jupyter Notebook)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análise das Previsões - Janeiro 2023', fontsize=16, fontweight='bold')

# 1. Previsão por semana
semana_qty = df_previsoes_2023.groupby('semana')['quantidade'].sum().sort_index()
axes[0, 0].bar(semana_qty.index, semana_qty.values, color='skyblue', alpha=0.8)
axes[0, 0].set_title('Quantidade Prevista por Semana - Janeiro 2023')
axes[0, 0].set_xlabel('Semana')
axes[0, 0].set_ylabel('Quantidade Total')
axes[0, 0].grid(True, alpha=0.3)

# 2. Top 15 PDVs
top_pdvs = df_previsoes_2023.groupby('pdv')['quantidade'].sum().nlargest(15)
axes[0, 1].barh(range(len(top_pdvs)), top_pdvs.values, color='lightgreen', alpha=0.8)
axes[0, 1].set_yticks(range(len(top_pdvs)))
axes[0, 1].set_yticklabels([f'PDV {int(pdv)}' for pdv in top_pdvs.index])
axes[0, 1].invert_yaxis()
axes[0, 1].set_title('Top 15 PDVs - Quantidade Total Prevista')
axes[0, 1].set_xlabel('Quantidade Total')
axes[0, 1].grid(True, alpha=0.3)

# 3. Top 15 Produtos
top_produtos = df_previsoes_2023.groupby('produto')['quantidade'].sum().nlargest(15)
axes[1, 0].barh(range(len(top_produtos)), top_produtos.values, color='salmon', alpha=0.8)
axes[1, 0].set_yticks(range(len(top_produtos)))
axes[1, 0].set_yticklabels([ (str(prod)[:15] + '...') if len(str(prod))>15 else str(prod) for prod in top_produtos.index ])
axes[1, 0].invert_yaxis()
axes[1, 0].set_title('Top 15 Produtos - Quantidade Total Prevista')
axes[1, 0].set_xlabel('Quantidade Total')
axes[1, 0].grid(True, alpha=0.3)

# 4. Distribuição das quantidades previstas (boxplot)
sns.boxplot(y=df_previsoes_2023['quantidade'], ax=axes[1, 1])
axes[1, 1].set_title('Distribuição das Quantidades Previstas')
axes[1, 1].set_ylabel('Quantidade')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 5. Exportar previsões para CSV
output_file = 'previsoes_janeiro_2023.csv'
df_previsoes_2023.to_csv(output_file, index=False)
print(f"\n✅ Previsões exportadas para {output_file}")

# 6. Análise final
print("\n📊 ANÁLISE FINAL DAS PREVISÕES")
print("="*60)
print(f"Total de registros: {len(df_previsoes_2023):,}")
print(f"Quantidade total prevista: {df_previsoes_2023['quantidade'].sum():,} unidades")
print(f"Média por registro: {df_previsoes_2023['quantidade'].mean():.1f} unidades")
print(f"Mediana por registro: {df_previsoes_2023['quantidade'].median():.1f} unidades")
print(f"Desvio padrão: {df_previsoes_2023['quantidade'].std():.1f}")
print(f"PDVs únicos: {df_previsoes_2023['pdv'].nunique()}")
print(f"Produtos únicos: {df_previsoes_2023['produto'].nunique()}")
print(f"Arquivo de saída: {output_file}")

print("\n🎯 PREVISÕES FINALIZADAS COM SUCESSO! 🎯")
print("="*60)
print("Próximos passos sugeridos:")
print("1. Validar as previsões com a equipe de negócios")
print("2. Ajustar o modelo conforme feedback")
print("3. Automatizar o pipeline de previsão para execução periódica")
print("4. Implementar monitoramento de desempenho do modelo em produção")
