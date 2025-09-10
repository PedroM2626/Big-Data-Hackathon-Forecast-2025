# 📂 Pasta de Dados Originais (Raw)

## ⚠️ DADOS NÃO INCLUÍDOS NO REPOSITÓRIO

Os arquivos de dados originais **não estão incluídos neste repositório** devido ao seu tamanho (arquivos parquet grandes).

## 📥 Como Obter os Dados

### 1. **Faça o Download:**
- **Acesse**: https://hackathon.bdtech.ai/download
- **Baixe** todos os arquivos no formato `.parquet`

### 2. **Coloque os Arquivos Aqui:**
- Salve todos os arquivos `.parquet` baixados **nesta pasta** (`data/raw/`)

### 3. **Estrutura Esperada:**
Após o download, esta pasta deve conter:
```
data/raw/
├── part-00000-tid-[número1].snappy.parquet
├── part-00000-tid-[número2].snappy.parquet 
├── part-00000-tid-[número3].snappy.parquet
└── ... (outros arquivos parquet)
```

## 📋 Arquivos Esperados

O sistema de forecasting espera encontrar arquivos com padrão:
- **Formato**: `.parquet` ou `.snappy.parquet`
- **Nomenclatura**: `part-00000-tid-*`
- **Conteúdo**: Dados históricos de vendas de 2022

## 🔄 Fallback Automático

**Não se preocupe!** Se os arquivos não forem encontrados:
- ✅ O sistema **automaticamente** usa dados simulados realistas
- ✅ Todas as funcionalidades continuam funcionando
- ✅ As previsões são geradas normalmente

## 🎯 Próximos Passos

1. **Baixe** os dados do link acima
2. **Coloque** os arquivos nesta pasta
3. **Execute** o notebook ou script principal
4. **Verifique** se os dados reais são carregados nos logs

---

**💡 Dica**: Se aparecer "🔄 Utilizando dados simulados" nos logs, significa que os arquivos parquet não foram encontrados ou não puderam ser carregados.
