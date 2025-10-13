import os
import shutil
import torch
import pickle
import re
import pymupdf  # Usaremos o PyMuPDF diretamente para controle total
from tqdm import tqdm
from langchain_core.documents import Document
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever

# --- PARÂMETROS ---
DOCS_PATH = "documentos/"
VECTORSTORE_PATH = "faiss_final_index"
DOCSTORE_PATH = "final_docstore"
MODEL_EMBEDDING_NAME = "BAAI/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cleanup_old_index(vectorstore_path, docstore_path):
    """Limpa os diretórios antigos do índice e do docstore."""
    for path in [vectorstore_path, docstore_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(vectorstore_path, exist_ok=True)
    os.makedirs(docstore_path, exist_ok=True)
    print("Limpeza e recriação das pastas concluída.")


def load_and_clean_definitively(folder_path: str) -> list:
    """
    Solução definitiva: Extrai blocos de texto, ignora cabeçalhos/rodapés,
    ordena pela ordem de leitura, une o conteúdo e normaliza os espaços.
    """
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    all_cleaned_pages = []

    for filename in tqdm(pdf_files, desc="Processando PDFs de forma definitiva"):
        file_path = os.path.join(folder_path, filename)

        with pymupdf.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                page_height = page.rect.height
                margin_top = page_height * 0.08
                margin_bottom = page_height * 0.92

                blocks = page.get_text("blocks")
                valid_blocks = [
                    b for b in blocks if margin_top < b[1] < margin_bottom]
                valid_blocks.sort(key=lambda b: b[1])

                page_text = " ".join([b[4].replace('\n', ' ')
                                     for b in valid_blocks])
                all_cleaned_pages.append(page_text)

    # Une o texto de todas as páginas limpas em um único grande documento
    full_text_content = " ".join(all_cleaned_pages)

    # ## --- LINHAS ADICIONADAS PARA LIMPEZA DEFINITIVA --- ##
    # Substitui qualquer sequência de espaços, tabulações ou quebras de linha por um único espaço
    full_text_content = re.sub(r'\s+', ' ', full_text_content).strip()
    # ## --------------------------------------------------- ##

    print(
        f"Limpeza concluída. Conteúdo total com {len(full_text_content)} caracteres.")
    return [Document(page_content=full_text_content)]


def main():
    cleanup_old_index(VECTORSTORE_PATH, DOCSTORE_PATH)

    docs = load_and_clean_definitively(DOCS_PATH)

    # Splitters para a estratégia Parent/Child
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200, add_start_index=True)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50)

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_EMBEDDING_NAME, model_kwargs={'device': DEVICE})

    vectorstore = FAISS.from_texts(
        texts=["_INITIALIZING_"], embedding=embeddings)
    vectorstore.delete(list(vectorstore.index_to_docstore_id.values()))

    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore, docstore=store,
        child_splitter=child_splitter, parent_splitter=parent_splitter,
    )

    print("Adicionando documento limpo ao índice...")
    retriever.add_documents(docs, ids=None)

    vectorstore.save_local(VECTORSTORE_PATH)
    with open(os.path.join(DOCSTORE_PATH, "store.pkl"), "wb") as f:
        pickle.dump(store, f)

    print(
        f"Índice definitivo salvo em '{VECTORSTORE_PATH}' e '{DOCSTORE_PATH}'")


if __name__ == "__main__":
    main()
