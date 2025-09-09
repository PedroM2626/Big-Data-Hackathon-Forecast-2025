# Contribuindo para o Projeto Big Data Hackathon Forecast 2025

Obrigado pelo seu interesse em contribuir para o nosso projeto! Este guia fornecerá informações sobre como você pode ajudar a melhorar nossa solução de previsão de vendas.

## Como Contribuir

### 1. Configuração do Ambiente

1. **Faça um Fork** do repositório
2. **Clone** o repositório para sua máquina local:
   ```bash
   git clone https://github.com/seu-usuario/big-data-hackathon-forecast-2025.git
   cd big-data-hackathon-forecast-2025
   ```

3. **Crie um ambiente virtual** (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

4. **Instale as dependências**:
   ```bash
   pip install -e ".[dev]"  # Instala o pacote em modo desenvolvimento com dependências extras
   ```

5. **Configure as variáveis de ambiente**:
   ```bash
   cp .env.example .env
   # Edite o arquivo .env conforme necessário
   ```

### 2. Convenções de Código

- Siga o [PEP 8](https://www.python.org/dev/peps/pep-0008/) para estilo de código Python
- Use docstrings seguindo o formato [Google Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Escreva testes para novas funcionalidades
- Documente todas as funções e classes públicas

### 3. Fluxo de Trabalho

1. Crie uma branch para sua feature/correção:
   ```bash
   git checkout -b feature/nome-da-feature
   # ou
   git checkout -b fix/corrigir-bug
   ```

2. Faça commit das suas alterações com mensagens descritivas:
   ```bash
   git commit -m "Adiciona funcionalidade X"
   ```

3. Envie suas alterações para o repositório remoto:
   ```bash
   git push origin feature/nome-da-feature
   ```

4. Abra um Pull Request (PR) para a branch `main`

### 4. Testes

Certifique-se de que todos os testes passam antes de enviar um PR:

```bash
pytest tests/
```

### 5. Padrões de Commit

Utilize o seguinte padrão para mensagens de commit:

```
tipo(escopo): descrição concisa

Corpo mais detalhado, se necessário

[OPCIONAL] Rodapé com referências a issues, etc.
```

Tipos de commit:
- `feat`: Nova funcionalidade
- `fix`: Correção de bug
- `docs`: Alterações na documentação
- `style`: Formatação, ponto e vírgula, etc. (sem mudança de código)
- `refactor`: Refatoração de código
- `test`: Adicionando testes
- `chore`: Atualização de tarefas de build, configuração, etc.

### 6. Diretrizes para Pull Requests

- Mantenha os PRs pequenos e focados em uma única funcionalidade/correção
- Inclua uma descrição clara do que foi feito e por quê
- Referencie as issues relacionadas
- Certifique-se de que todos os testes estão passando
- Atualize a documentação conforme necessário

### 7. Reportando Problemas

Ao reportar um bug, por favor inclua:
- Descrição clara do problema
- Passos para reproduzir
- Comportamento esperado vs. comportamento atual
- Capturas de tela, se aplicável
- Versão do Python e das dependências

### 8. Código de Conduta

Este projeto segue o [Código de Conduta do Contribuidor](CODE_OF_CONDUCT.md). Ao participar, espera-se que você siga este código.

## Agradecimentos

Agradecemos a todos os contribuidores que ajudam a melhorar este projeto!
