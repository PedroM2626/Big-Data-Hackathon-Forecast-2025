# 🎯 Big Data Hackathon Forecast 2025

## Modelo de Previsão de Vendas para Sistema One-Click Order

Este projeto desenvolve um sistema de forecasting para apoiar a reposição automática de estoque, prevendo quantidades semanais de vendas por PDV/SKU para otimizar o processo de compras.

## 📋 Objetivo

Desenvolver um modelo de machine learning para prever a quantidade semanal de vendas por **PDV (Ponto de Venda)** e **SKU (Stock Keeping Unit)** para as 5 primeiras semanas de janeiro de 2023, baseado nos dados históricos de vendas de 2022.

## 📊 Dados

- **Período de Análise**: Histórico de vendas de 2022
- **Período de Previsão**: 5 semanas de janeiro de 2023
- **Granularidade**: Semanal por PDV e SKU
- **Formato**: Arquivos Parquet na pasta `data/raw/`

## 🔧 Tecnologias Utilizadas

- **Python 3.8+**
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica  
- **Scikit-learn** - Modelos de machine learning
- **Matplotlib/Seaborn** - Visualizações
- **Jupyter Notebook** - Análise interativa
- **PyArrow** - Leitura de arquivos Parquet

## 📁 Estrutura do Projeto

```
Big-Data-Hackathon-Forecast-2025/
├── data/
│   ├── raw/                           # Dados brutos (arquivos parquet)
│   └── processed/                     # Dados processados e previsões
├── tests/                             # Scripts Python de teste e desenvolvimento
│   ├── forecast_demo_complete.py      # ⭐ Script principal funcional
│   ├── simple_test.py                 # Teste básico de bibliotecas
│   └── README.md                      # Documentação dos scripts
├── forecast_model_notebook.ipynb      # 📓 Notebook principal
├── requirements.txt                   # Dependências do projeto
├── Makefile                          # Comandos automatizados
└── README.md                         # Este arquivo
```

## 📥 Preparação dos Dados

**⚠️ IMPORTANTE**: Os dados originais não estão incluídos no repositório devido ao tamanho.

### Download dos Dados:
1. **Acesse**: https://hackathon.bdtech.ai/download
2. **Baixe** os arquivos que estão no formato `.parquet`
3. **Coloque** os arquivos baixados na pasta `data/raw/`

A estrutura deve ficar assim:
```
data/raw/
├── part-00000-tid-xxxxx.snappy.parquet
├── part-00000-tid-yyyyy.snappy.parquet
└── part-00000-tid-zzzzz.snappy.parquet
```

## 🚀 Como Executar

### Opção 1: Jupyter Notebook (Recomendado para Análise)
```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Abrir Jupyter Notebook
jupyter notebook forecast_model_notebook.ipynb

# 3. Executar células sequencialmente (Shift+Enter)
```

### Opção 2: Script Python (Execução Direta)
```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar script principal
python tests/forecast_demo_complete.py
```

### Opção 3: Testes Básicos
```bash
# Testar funcionalidades básicas
python tests/simple_test.py

# Testar carregamento de dados
python tests/test_data_loading.py
```

## 📈 Pipeline de Machine Learning

1. **📂 Carregamento de Dados**: Arquivos parquet ou dados simulados
2. **📊 Análise Exploratória**: Padrões, sazonalidade e tendências
3. **⚙️ Feature Engineering**: Variáveis temporais e categóricas
4. **🤖 Modelagem**: Random Forest Regressor otimizado
5. **✅ Validação**: Split temporal (últimas 8 semanas)
6. **🎯 Previsões**: Forecasts para 5 semanas de janeiro 2023
7. **💾 Exportação**: CSV e relatórios em `data/processed/`

## 📊 Métricas de Avaliação

- **MAE** (Mean Absolute Error) - Erro médio absoluto
- **RMSE** (Root Mean Square Error) - Raiz do erro quadrático médio  
- **MAPE** (Mean Absolute Percentage Error) - Erro percentual médio
- **R²** (Coeficiente de Determinação) - Qualidade do ajuste

## 📈 Resultados do Modelo

### Exemplo de Performance:
- **MAE**: ~5.48 unidades
- **RMSE**: ~7.61 unidades  
- **R²**: ~0.25
- **MAPE**: ~25%

### Saídas Geradas:
- **`data/processed/previsoes_janeiro_2023_[timestamp].csv`**
  ```
  semana,pdv,produto,quantidade
  1,1023,SKU_1001,15
  1,1023,SKU_1002,23
  ...
  ```
- **`data/processed/resumo_forecasting_[timestamp].txt`** - Relatório executivo

## 🎯 Features do Sistema

✅ **Fallback Automático**: Usa dados simulados se parquet falhar  
✅ **Validação Temporal**: Evita data leakage  
✅ **Visualizações Integradas**: Gráficos de análise e resultados  
✅ **Export Automático**: CSV e relatórios estruturados  
✅ **Documentação Completa**: Notebooks com explicações detalhadas  
✅ **Testes Validados**: Scripts testados e funcionais  

## 🛠️ Troubleshooting

### Problemas Comuns:
1. **Erro com arquivos parquet**: O sistema usa dados simulados automaticamente
2. **Bibliotecas não encontradas**: Execute `pip install -r requirements.txt`
3. **Jupyter não abre**: Instale com `pip install jupyter notebook`

### Para Suporte:
- Verifique `tests/simple_test.py` para validar bibliotecas
- Use `tests/forecast_demo_complete.py` se o notebook falhar
- Consulte `tests/README.md` para detalhes dos scripts

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)  
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 👥 Equipe de Desenvolvimento

**Projeto desenvolvido por:**
- **Pedro Morato**
- **Pietra Paz** 
- **Erick Mendes**

## 🏆 Hackathon

**Big Data Hackathon 2025** - Sistema One-Click Order

---

**Status**: ✅ **Funcional e Testado** 🚀
