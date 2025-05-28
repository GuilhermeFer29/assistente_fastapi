import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir)) # Sobe um nível para chegar na pasta raiz do projeto
sys.path.append(project_root)

import streamlit as st
from src.llm_interactions import ask_question
from src.config import APP_TITLE, APP_DESCRIPTION
import os

st.set_page_config(page_title=APP_TITLE, layout="centered")

st.title(APP_TITLE)
st.markdown(APP_DESCRIPTION)

# Botão para instruir o usuário a popular o DB
st.warning("Se esta é a primeira vez rodando ou se sua base de conhecimento mudou, execute 'python scripts/populate_vector_db.py' no terminal.")

# Histórico de conversas (opcional, para persistência no Streamlit)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Opção de idioma
idioma_selecionado = st.selectbox(
    "Escolha o idioma da resposta:",
    ("Português do Brasil", "English"),
    index=0 # Português como padrão
)

# Entrada do usuário
if user_question := st.chat_input("Pergunte algo sobre programação..."):
    # Adiciona a pergunta do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Gera a resposta do assistente
    with st.chat_message("assistant"):
        with st.spinner("Pensando na resposta..."):
            response_content = ask_question(user_question, idioma_desejado=idioma_selecionado)
            st.markdown(response_content)
        st.session_state.messages.append({"role": "assistant", "content": response_content})