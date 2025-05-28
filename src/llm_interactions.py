import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from operator import itemgetter 
from langchain_community.vectorstores import FAISS 
import src.config as config

def get_llm():
    """
    Retorna o modelo de linguagem grande (LLM) a ser usado, configurado para OpenRouter.
    """
    if config.OPENROUTER_API_KEY and config.OPENROUTER_BASE_URL:
        print(f"Usando OpenRouter com modelo: {config.LLM_MODEL_NAME}")
        return ChatOpenAI(
            model_name=config.LLM_MODEL_NAME,
            temperature=config.TEMPERATURE,
            openai_api_key=config.OPENROUTER_API_KEY,
            openai_api_base=config.OPENROUTER_BASE_URL
        )
    else:
        print("OPENROUTER_API_KEY ou OPENROUTER_BASE_URL não configurados. Não foi possível inicializar LLM.")
        print("Considere usar um modelo open-source ou configurar a chave OpenRouter.")
        return None


def get_embeddings_model():
    """
    Retorna o modelo de embeddings a ser usado (OpenAI ou HuggingFace).
    """
    if config.OPENAI_API_KEY:
        print("Usando OpenAIEmbeddings para embeddings...")
        return OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    else:
        print("OPENAI_API_KEY não configurada para embeddings. Usando HuggingFaceEmbeddings (modelo all-MiniLM-L6-v2)...")

        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 


def get_retriever():
    """
    Carrega o banco de dados vetorial (FAISS) e configura o retriever.
    """
    embeddings_model = get_embeddings_model()

    # Verifica se o arquivo FAISS existe
    if not os.path.exists(config.VECTOR_DB_PATH):
        print(f"Erro: Banco de dados vetorial FAISS não encontrado em {config.VECTOR_DB_PATH}. Por favor, execute populate_vector_db.py primeiro.")
        return None

    print(f"Carregando banco de dados vetorial FAISS de: {config.VECTOR_DB_PATH}")
    
    # Carrega o FAISS
    db = FAISS.load_local(
        folder_path=os.path.dirname(config.VECTOR_DB_PATH),
        embeddings=embeddings_model,
        index_name=os.path.basename(config.VECTOR_DB_PATH).split('.')[0],
        allow_dangerous_deserialization=True 
    )
    return db.as_retriever(search_kwargs={"k": config.TOP_K_RETRIEVER})


def get_qa_chain():
    """
    Cria e retorna a cadeia de Q&A usando LangChain Expression Language (LCEL).
    """
    llm = get_llm()
    retriever = get_retriever()

    if llm is None or retriever is None:
        print("Não foi possível inicializar LLM ou Retriever. Verifique suas configurações e se o DB foi populado.")
        return None

    # --- Prompt para o LLM ---
    template = """Você é um assistente de programação prestativo.
    Use o seguinte contexto para responder à pergunta.
    Se a resposta não estiver no contexto, diga que não sabe.
    Sua tarefa é responder à pergunta do usuário **EXCLUSIVAMENTE no idioma solicitado**.
    Responda em {idioma_desejado}.

    Contexto: {context}

    Pergunta: {question}

    Resposta:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Função para formatar os documentos recuperados em uma única string de contexto
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # --- Construção da Cadeia com LCEL ---
    qa_chain = (
        {
            
            "context": itemgetter("query") | retriever | RunnableLambda(format_docs),
           
            "question": itemgetter("query"),
            
            "idioma_desejado": itemgetter("idioma_desejado"),
        }
        | QA_CHAIN_PROMPT
        | llm
    )
    return qa_chain

def ask_question(question: str, idioma_desejado: str = "Português do Brasil"):
    """
    Faz uma pergunta ao assistente de programação e retorna a resposta e as fontes.
    """
    qa_chain = get_qa_chain()
    if qa_chain is None:
        return "Desculpe, o assistente não está configurado corretamente. Verifique o console para erros."


    result_dict = qa_chain.invoke({"query": question, "idioma_desejado": idioma_desejado})
    

    response_content = result_dict.content if hasattr(result_dict, 'content') else str(result_dict)



    return response_content # Por enquanto, sem fontes com esta estrutura LCEL simplificada.