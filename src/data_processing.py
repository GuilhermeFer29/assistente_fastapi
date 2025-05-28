import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import src.config as config


def get_documents_from_source():
    """
    Carrega documentos de diferentes formatos da pasta DOCS_PATH.
    """
    print(f"Carregando documentos da pasta: {config.DOCS_PATH}") # Usar config.DOCS_PATH
    documents = []

    pdf_loader = DirectoryLoader(
        config.DOCS_PATH, # Usar config.DOCS_PATH
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        use_multithreading=True
    )
    documents.extend(pdf_loader.load())

    txt_loader = DirectoryLoader(
        config.DOCS_PATH, # Usar config.DOCS_PATH
        glob="**/*.txt",
        loader_cls=TextLoader,
        use_multithreading=True
    )
    documents.extend(txt_loader.load())

    if not documents:
        print("Nenhum documento encontrado para processar.")
        return []

    print(f"Total de {len(documents)} documentos carregados.")
    return documents

def split_documents_into_chunks(documents: list[Document]):
    """
    Divide uma lista de documentos em chunks menores.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, # Usar config.CHUNK_SIZE
        chunk_overlap=config.CHUNK_OVERLAP, # Usar config.CHUNK_OVERLAP
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documentos divididos em {len(chunks)} chunks.")
    return chunks

def get_embeddings_model():
    """
    Retorna o modelo de embeddings a ser usado (OpenAI ou HuggingFace).
    """
    if config.OPENAI_API_KEY: # Acessar via config.OPENAI_API_KEY
        print("Usando OpenAIEmbeddings para embeddings...")
        return OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    else:
        print("OPENAI_API_KEY não configurada para embeddings. Usando HuggingFaceEmbeddings (modelo all-MiniLM-L6-v2)...")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_and_store_documents():
    """
    Orquestra o carregamento, divisão, embedding e armazenamento dos documentos no FAISS.
    """
    documents = get_documents_from_source()
    if not documents:
        print("Não há documentos para processar. Saindo.")
        return

    chunks = split_documents_into_chunks(documents)
    embeddings_model = get_embeddings_model()

    # FAISS 
    if os.path.exists(config.VECTOR_DB_PATH):
        print(f"Removendo banco de dados vetorial existente em: {config.VECTOR_DB_PATH}")
        os.remove(config.VECTOR_DB_PATH) 

        if os.path.exists(config.VECTOR_DB_PATH.replace('.faiss', '.pkl')):
             os.remove(config.VECTOR_DB_PATH.replace('.faiss', '.pkl'))


    print("Criando e populando o banco de dados vetorial (FAISS)...")
    db = FAISS.from_documents(chunks, embeddings_model)
    db.save_local(os.path.dirname(config.VECTOR_DB_PATH), os.path.basename(config.VECTOR_DB_PATH).split('.')[0]) # Salva o arquivo FAISS

    print(f"Banco de dados vetorial criado e populado em: {config.VECTOR_DB_PATH}")
    print("Processo de populamento concluído com sucesso!")