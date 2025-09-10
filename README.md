# Big Data Hackathon Forecast 2025

## Visão do Projeto
Este projeto tem como objetivo prever a quantidade semanal de vendas por PDV (Ponto de Venda) e SKU (Stock Keeping Unit) para as quatro primeiras semanas de janeiro de 2023, utilizando dados históricos de vendas de 2022.

## Equipe
- Pedro Morato
- Pietra Paz
- Erick Mendes

## Estrutura do Projeto
```
Big-Data-Hackathon-Forecast-2025/
├── data/                # Dados do projeto
│   ├── raw/            # Dados brutos (a serem baixados)
│   └── processed/      # Dados processados
├── LICENSE             # Licença do projeto
├── README.md           # Documentação principal
├── Makefile            # Comandos úteis para o projeto
├── products_recommendation_notebook.ipynb  # Notebook de análise
└── requirements.txt    # Dependências do projeto
```

## Como Executar

### Pré-requisitos
- Python 3.8+
- Git
- Make (opcional, mas recomendado)

### Configuração Inicial

1. **Clone o repositório**
   ```bash
   git clone https://github.com/PedroM2626/Big-Data-Hackathon-Forecast-2025.git
   cd Big-Data-Hackathon-Forecast-2025
   ```

2. **Baixe os dados**
   - Acesse https://hackathon.bdtech.ai/download
   - Baixe os arquivos que estão no formato .parquet
   - Crie a pasta `data/raw` se ela não existir
   - Coloque os arquivos baixados na pasta `data/raw/`

3. **Configure o ambiente**
   ```bash
   # Crie e ative o ambiente virtual (recomendado)
   make setup
   
   # No Windows:
   .\venv\Scripts\activate
   
   # No Linux/Mac:
   # source venv/bin/activate
   ```

4. **Instale as dependências**
   ```bash
   make install
   ```

### Uso

#### Com Makefile (recomendado)
```bash
# Iniciar o Jupyter Notebook
make notebook

# Formatar o código
make format

# Verificar qualidade do código
make lint

# Limpar arquivos temporários
make clean
```

#### Sem Makefile
```bash
# Instalar dependências
pip install -r requirements.txt

# Iniciar o Jupyter Notebook
jupyter notebook products_recommendation_notebook.ipynb
```

## Comandos Úteis do Makefile

- `make setup`: Cria o ambiente virtual e instala as dependências
- `make install`: Instala as dependências do projeto
- `make notebook`: Inicia o Jupyter Notebook
- `make format`: Formata o código automaticamente
- `make lint`: Verifica a qualidade do código
- `make clean`: Remove arquivos temporários
- `make help`: Mostra todos os comandos disponíveis

## Estrutura de Dados

Os dados devem ser organizados da seguinte forma:

```
data/
├── raw/               # Dados brutos (não versionados no git)
│   ├── dados que foram baixados
│   ├── dados que foram baixados
│   └── dados que foram baixados
└── processed/         # Dados processados (gerados pelo notebook)
    ├── dados_tratados.parquet
    └── previsoes.parquet
```

## Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Faça commit das suas alterações (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.
