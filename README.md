
# Assistente FastAPI: Seu Guia Inteligente para Dúvidas sobre FastAPI com RAG

Este projeto é um assistente inteligente especializado em FastAPI que utiliza a tecnologia de Geração Aumentada por Recuperação (RAG) e Modelos de Linguagem Grandes (LLMs) para responder a perguntas técnicas com base em uma base de conhecimento personalizada e local. O objetivo é fornecer respostas rápidas e contextuais sobre o framework FastAPI, tirando dúvidas de documentações e tutoriais.

## 💡 Problema Resolvido

Programadores frequentemente gastam tempo valioso buscando informações específicas em documentações extensas do FastAPI ou em fóruns. Este assistente centraliza o conhecimento sobre FastAPI, permitindo respostas instantâneas e contextualizadas a partir de uma base de dados interna e confiável, otimizando o fluxo de trabalho dos desenvolvedores.

## ✨ Funcionalidades Principais

- **Q&A em Linguagem Natural**: Faça perguntas sobre a utilização do framework FastAPI em português ou inglês e receba respostas claras.
- **Base de Conhecimento Especializada**: Alimentado pela documentação oficial do FastAPI e recursos selecionados, garantindo respostas relevantes e precisas.
- **Geração Aumentada por Recuperação (RAG)**: O sistema recupera informações relevantes da base de conhecimento antes de gerar a resposta com o LLM, garantindo precisão e reduzindo "alucinações".
- **Suporte a Documentos Bilíngues**: Processa documentos em português e inglês.
- **Resposta no Idioma Desejado**: O usuário pode selecionar o idioma da resposta diretamente na interface.
- **Interface Profissional**: Interface web moderna e amigável construída com Streamlit, com exibição apropriada do histórico de chat.
- **Uso de LLM Avançado**: Integração com o modelo DeepSeek-chat-v3-0324 via OpenRouter.

## 🛠️ Tecnologias Utilizadas

- **Linguagem de Programação**: Python
- **Frameworks/Bibliotecas**:
  - FastAPI
  - LangChain
  - Streamlit
  - HuggingFaceEmbeddings
  - ChromaDB
  - pypdf / unstructured
  - OpenRouter (DeepSeek-chat-v3-0324)
  - python-dotenv
  - os, sys, shutil
- **Conceitos**: RAG, Embeddings, LLMs, PLN, Vetores, API REST

## 📁 Estrutura do Projeto

```
assistente_fastapi/
├── .venv/
├── docs/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── data_processing.py
│   ├── api.py
│   └── llm_interactions.py
├── scripts/
│   └── populate_vector_db.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Como Rodar o Projeto (Localmente)

### Pré-requisitos

- Python 3.8+ instalado.
- Git instalado.

### 1. Clonar o Repositório

```bash
git clone https://github.com/GuilhermeFer29/assistente_fastapi.git
cd assistente_fastapi
```

### 2. Configurar o Ambiente Virtual

```bash
python -m venv .venv
```

Ativar o ambiente virtual:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate

# Windows CMD
.venv\Scripts\activate.bat
```

### 3. Instalar as Dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar Variáveis de Ambiente

Crie um arquivo `.env` e adicione sua chave:

```env
OPENROUTER_API_KEY="SUA_CHAVE_OPENROUTER_AQUI"
# OPENAI_API_KEY="SUA_CHAVE_OPENAI_PARA_EMBEDDINGS_SE_FOR_USAR_AQUI"
```

### 5. Adicionar Documentos à Pasta `docs/`

Inclua seus PDFs e TXTs na pasta `docs/`.

### 6. Popular o Banco de Dados Vetorial

```bash
python scripts/populate_vector_db.py
```

### 7. Iniciar o Aplicativo

```bash
streamlit run src/main.py
```

## 🚀 Uso do Assistente

Escolha o idioma, digite sua pergunta sobre FastAPI, e receba a resposta contextualizada no idioma desejado. O histórico de chat será exibido de forma clara e organizada na interface.

## 🚧 Desafios e Aprendizados

- Organização modular de projetos Python
- Implementação do pipeline RAG
- Uso de embeddings sem custo (HuggingFace)
- Integração com LLMs via OpenRouter
- Controle de idioma nas respostas
- Design de interfaces profissionais com Streamlit
- Implementação de APIs REST com FastAPI

## 🎯 Próximos Passos

- Suporte a DOCX, HTML
- Cache de embeddings
- Expansão da base de conhecimento com mais recursos de FastAPI
- Fine-tuning de LLMs específico para FastAPI
- Deploy em nuvem
- Integração com ambientes de desenvolvimento (VSCode, PyCharm)

## 👨‍💻 Autor

**Guilherme Fernandes do Bem**  
[LinkedIn](https://linkedin.com/in/guilherme-fernandes-do-bem) 