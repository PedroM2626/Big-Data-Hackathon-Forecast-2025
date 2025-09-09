# Big Data Hackathon Forecast 2025

## Visão do Projeto
Este projeto tem como objetivo prever a quantidade semanal de vendas por PDV (Ponto de Venda) e SKU (Stock Keeping Unit) para as quatro primeiras semanas de janeiro de 2023, utilizando dados históricos de vendas de 2022.

## Equipe
- Pedro Morato
- Pietra Paz
- Alisson Guarniêr

## Estrutura do Projeto
```
Big-Data-Hackathon-Forecast-2025/
├── data/               # Dados brutos e processados
├── docs/               # Documentação do projeto
├── models/             # Modelos treinados
├── notebooks/          # Jupyter notebooks para análise exploratória
└── src/                # Código-fonte
    ├── data/           # Scripts para processamento de dados
    ├── features/       # Engenharia de features
    ├── models/         # Definição e treinamento de modelos
    └── utils/          # Utilitários gerais
```

## Pipeline de Dados
1. **Extração**: Carregamento dos dados de vendas
2. **Limpeza**: Tratamento de valores ausentes e outliers
3. **Feature Engineering**: Criação de novas variáveis
4. **Modelagem**: Treinamento do modelo de previsão
5. **Avaliação**: Métricas de desempenho do modelo
6. **Previsão**: Geração das previsões finais

## Como Executar
1. Clone o repositório
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute o pipeline: `python src/main.py`

## Próximos Passos
- [ ] Coletar e organizar os dados de vendas
- [ ] Implementar pipeline de processamento
- [ ] Desenvolver modelo de previsão
- [ ] Validar resultados
- [ ] Otimizar desempenho
