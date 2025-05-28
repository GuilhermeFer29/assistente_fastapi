import os
import sys
# Isso permite que o script encontre o pacote 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
from src.data_processing import process_and_store_documents

if __name__ == "__main__":
    print("Iniciando o processo de populamento do banco de dados vetorial...")
    process_and_store_documents()
    print("Processo concluído.")