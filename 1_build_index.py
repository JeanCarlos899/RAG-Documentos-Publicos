import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
DOCS_PATH = "documentos/"
INDEX_PATH = "faiss_index"
MODEL_NAME = "BAAI/bge-m3"

def load_documents(folder_path):
    """Carrega todos os PDFs de uma pasta e os retorna como uma lista de documentos."""
    if not os.path.exists(folder_path):
        print(f"A pasta '{folder_path}' não foi encontrada.")
        return []
        
    all_docs = []
    print("Carregando documentos...")
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyMuPDFLoader(file_path)
            all_docs.extend(loader.load())
    return all_docs

def split_documents(docs):
    """Divide os documentos em chunks menores."""
    print("Dividindo documentos em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return text_splitter.split_documents(docs)

def create_embeddings(model_name):
    """Inicializa o modelo de embedding para rodar na GPU (cuda)."""
    print("Inicializando modelo de embedding...")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

def create_and_save_index(chunks, embeddings, index_path):
    """Cria o índice FAISS a partir dos chunks e o salva no disco."""
    print("Criando e salvando o índice FAISS... (isso pode levar um tempo)")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"Índice salvo com sucesso em '{index_path}'")

def main():
    documents = load_documents(DOCS_PATH)
    if not documents:
        print("Nenhum documento para processar. Encerrando.")
        return
        
    chunks = split_documents(documents)
    embeddings = create_embeddings(MODEL_NAME)
    create_and_save_index(chunks, embeddings, INDEX_PATH)

if __name__ == "__main__":
    main()