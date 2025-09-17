# 🎯 Big Data Forecast Hackathon Forecast 2025

## Modelo de Previsão de Vendas para Sistema One-Click Order

Este projeto desenvolve um sistema de forecasting para apoiar a reposição automática de estoque, prevendo quantidades semanais de vendas por PDV/SKU para otimizar o processo de compras.

## 🌟 Novidades da Versão 2.0

- **Eficiência Aprimorada**: O indicador de eficiência (bins) saltou de **1.400 para 14.100**, representando um ganho massivo na capacidade de processamento e análise.
- **Artefatos de Treinamento Completos**: Feedbacks de treinamento muito mais detalhados e completos, permitindo uma análise mais profunda do desempenho do modelo.
- **Inteligência de Negócios Aplicada**: Melhoria crucial na representação das tabelas de dimensão, agora exibindo as informações essenciais para a tomada de decisão.
- **Foco no Notebook**: O arquivo `Forecast_Model_Notebook.ipynb` é agora o ponto central do projeto, simplificando a execução e análise.

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
├── artifacts/
│   └── sales_forecaster.joblib        # Artefato do modelo treinado
├── data/
│   └── raw/                           # Dados brutos (arquivos parquet)
├── .gitignore
├── Forecast_Model_Notebook.ipynb      # 📓 Notebook principal para execução e análise
├── forescast_model.py                 # Script com a lógica do modelo
├── LICENSE
├── Makefile
├── README.md                          # Este arquivo
└── requirements.txt                   # Dependências do projeto
```

## 📥 Preparação dos Dados

**⚠️ IMPORTANTE**: Os dados originais não estão incluídos no repositório devido ao tamanho.

### Download dos Dados:
1. **Acesse**: https://drive.google.com/drive/folders/1SIJvM5ZCZV_yVdD4TepnY2csAPXwIDkw?usp=sharing 
2. **Baixe** os arquivos que estão no formato `.parquet`
3. **Coloque** os arquivos baixados na pasta `data/raw/`

A estrutura deve ficar assim:
```
data/raw/
├── dim_pdvs.parquet
├── dim_produtos.parquet
└── fato_vendas.parquet
```

## 🚀 Como Executar

O projeto agora é centrado no Jupyter Notebook para facilitar a análise e a execução.

```bash
# 1. Crie um ambiente virtual e ative-o
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Abra o Jupyter Notebook
jupyter notebook Forecast_Model_Notebook.ipynb

# 4. Execute as células sequencialmente (Shift+Enter)
```

## 📈 Pipeline de Machine Learning

1. **📂 Carregamento de Dados**: Arquivos parquet da pasta `data/raw/`.
2. **📊 Análise Exploratória**: Padrões, sazonalidade e tendências.
3. **⚙️ Feature Engineering**: Variáveis temporais e categóricas.
4. **🤖 Modelagem**: Random Forest Regressor otimizado.
5. **✅ Validação**: Split temporal (últimas 8 semanas).
6. **🎯 Previsões**: Forecasts para 5 semanas de janeiro 2023.
7. **💾 Exportação**: CSV com as previsões e artefato do modelo.

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
- **`artifacts/sales_forecaster.joblib`**: Artefato do modelo treinado.
- **`previsoes_janeiro_2023_[timestamp].csv`**: Arquivo com as previsões.

## 🛠️ Troubleshooting

### Problemas Comuns:
1. **Erro com arquivos parquet**: Certifique-se de que os arquivos foram baixados e colocados na pasta `data/raw/`.
2. **Bibliotecas não encontradas**: Execute `pip install -r requirements.txt` no seu ambiente virtual.
3. **Jupyter não abre**: Instale com `pip install jupyter notebook`.

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
- **Erick Mendes**

**Grupo: **
- BSB Data 01

## 🏆 Hackathon

**Big Data Forecast Hackathon 2025** - Sistema One-Click Order

---

**Status**: ✅ **Versão 2.0 Lançada** 🚀
