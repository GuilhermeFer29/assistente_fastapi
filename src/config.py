import os
from dotenv import load_dotenv

load_dotenv()

#Configuração do OpenRouter
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Configuraçoes de pasta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_PATH = os.path.join(BASE_DIR, "docs")
VECTOR_DB_PATH = os.path.join(BASE_DIR, ".chroma_db")

# Configuraçoes RAG
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVER = 3

LLM_MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"
TEMPERATURE = 0.5

APP_TITLE ="Assistente FastAPI"
APP_DESCRIPTION = "Seu guia rápido para dúvidas sobre o FastAPI e seus recursos."
APP_VERSION = "1.0.0"