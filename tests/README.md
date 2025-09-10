# 🧪 Pasta de Testes e Scripts Auxiliares

Esta pasta contém scripts Python utilizados para desenvolvimento, teste e validação do modelo de forecasting.

## 📁 Arquivos Disponíveis

### Scripts de Teste
- **`simple_test.py`** - Teste básico de funcionalidade das bibliotecas
- **`test_data_loading.py`** - Teste de carregamento dos arquivos parquet

### Scripts de Desenvolvimento  
- **`forecast_demo_complete.py`** - **⭐ Script principal funcional**
  - Versão Python standalone completa do modelo
  - Executa todo o pipeline de forecasting
  - Gera previsões para janeiro 2023
  - **Uso:** `python tests/forecast_demo_complete.py`

### Scripts Auxiliares
- **`forecast_model_stable.py`** - Versão estável alternativa
- **`forecast_working_demo.py`** - Demo funcional com dados simulados  
- **`products_recommendation_notebook.py`** - Versão convertida do notebook original

## 🚀 Como Usar

### Executar o Modelo Completo:
```bash
cd E:\Projetos\Big-Data-Hackathon-Forecast-2025
python tests/forecast_demo_complete.py
```

### Testar Funcionalidades Básicas:
```bash
python tests/simple_test.py
```

### Testar Carregamento de Dados:
```bash
python tests/test_data_loading.py
```

## 📊 Saídas Esperadas

Os scripts principais geram:
- **Previsões CSV:** `data/processed/previsoes_janeiro_2023_[timestamp].csv`
- **Relatórios:** `data/processed/resumo_forecasting_[timestamp].txt`
- **Logs no console** com métricas e estatísticas

## 🔧 Dependências

Todos os scripts requerem as bibliotecas listadas em `requirements.txt`:
- pandas, numpy, scikit-learn
- matplotlib, seaborn 
- pyarrow (para arquivos parquet)

## 📋 Notas

- Os scripts têm fallback para dados simulados caso os arquivos parquet não carreguem
- **`forecast_demo_complete.py`** é a versão mais estável e recomendada
- Todos os scripts foram testados e validados
- Use o **notebook principal** (`forecast_model_notebook.ipynb`) para análise interativa
