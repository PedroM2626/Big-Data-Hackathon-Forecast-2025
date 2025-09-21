# 🚀 Modelo Preditivo de Vendas - Hackathon 2025

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Completo-success)

Este repositório contém a solução completa para o desafio de previsão de vendas do Hackathon 2025. O projeto implementa um pipeline de Machine Learning de ponta para prever a demanda semanal de produtos por ponto de venda, utilizando um modelo Gradient Boosting (LightGBM) meticulosamente otimizado para máxima precisão e robustez.

---

## 🎯 Objetivo do Projeto

O objetivo principal deste projeto é desenvolver um sistema de previsão de vendas (`forecast`) para as primeiras cinco semanas de 2023, com base no histórico de transações de 2022. A solução visa otimizar a reposição de estoque, minimizando rupturas e excessos, e fornecendo uma base de dados sólida para a tomada de decisões estratégicas da empresa.

---

## 🛠️ Metodologia Aplicada

A solução foi desenvolvida de forma iterativa, evoluindo de um modelo base para um pipeline sofisticado que incorpora as melhores práticas da indústria de Data Science:

1.  **Engenharia de Features Avançada:** Foram criadas mais de 20 features a partir dos dados brutos para capturar padrões complexos, incluindo:
    * **Lags Sazonais:** Vendas de semanas, meses e do ano anterior para informar o modelo sobre o comportamento histórico.
    * **Janelas Móveis:** Média, desvio padrão e máximo de vendas em diferentes janelas de tempo para identificar tendências e volatilidade.
    * **Features Cíclicas:** Decomposição da sazonalidade semanal usando seno e cosseno para um aprendizado mais eficaz.

2.  **Validação Robusta:** Foi implementada uma estratégia de **validação Hold-Out temporal**, separando as últimas semanas de 2022 para avaliar o modelo em um cenário que simula a previsão de dados futuros e desconhecidos, garantindo uma métrica de performance confiável (MAE).

3.  **Otimização de Hiperparâmetros de Alta Precisão:** O passo decisivo para a performance do modelo foi a utilização da biblioteca **Optuna**. Realizamos uma busca Bayesiana exaustiva com **100 iterações (`trials`)** para encontrar a combinação de hiperparâmetros do LightGBM que minimizasse o erro de previsão, resultando em um modelo final altamente especializado e ajustado para este dataset.

4.  **Estratégia de Submissão Preditiva:** O arquivo final respeita o limite de 1.5 milhão de linhas selecionando as combinações (PDV, Produto) com base no **maior potencial de vendas futuras previsto pelo próprio modelo otimizado**, uma abordagem proativa que foca nos produtos de maior impacto.

---

## 📂 Estrutura do Repositório

O projeto está organizado da seguinte forma para garantir modularidade e clareza:

```
/
├── artifacts/              # Pasta para salvar o modelo treinado (.joblib)
├── data/
│   ├── raw/                # Dados brutos de entrada (.parquet)
│   └── processed/          # Previsões finais geradas pelo script (.parquet)
├── forecaster_class.py     # Arquivo contendo a classe principal do pipeline (SalesForecasterV2)
├── train.py                # Script para treinar o modelo LightGBM com Optuna
├── predict.py              # Script para gerar a previsão final usando o modelo treinado
└── requirements.txt        # Arquivo com as dependências do projeto
```

---

## ⚙️ Configuração do Ambiente

Para replicar o ambiente de desenvolvimento e executar os scripts, siga os
