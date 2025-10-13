import pandas as pd
import os
import torch
import gc
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download
import evaluate
from sentence_transformers import CrossEncoder

# --- PARÂMETROS ---
VECTORSTORE_PATH = "faiss_final_index"
DOCSTORE_PATH = "final_docstore"
MODEL_EMBEDDING_NAME = "BAAI/bge-m3"

# Configuração do Modelo Llama 3.1
MODEL_REPO_ID = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
MODEL_BASENAME = "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"

LLM_N_GPU_LAYERS = -1
LLM_N_CTX = 8192
RETRIEVER_K = 10
TOP_K_AFTER_RERANK = 4
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


def clear_gpu_memory(*args):
    """Libera a memória da GPU."""
    print("\n--- Limpando a memória da GPU ---")
    for arg in args:
        try:
            del arg
        except NameError:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cache da GPU limpo.")


def load_local_llm(model_path):
    """Carrega o LLM Llama 3.1 com configurações otimizadas."""
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=LLM_N_GPU_LAYERS,
        n_ctx=LLM_N_CTX,
        max_tokens=512,
        repeat_penalty=1.15,
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        n_batch=512,
        f16_kv=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        verbose=False
    )
    return llm


def create_rag_chain(llm):
    """Cria o prompt específico para Llama 3.1."""
    template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Você é um assistente especializado em responder perguntas com base em documentos acadêmicos.\n\n"
        "INSTRUÇÕES:\n"
        "- Leia atentamente o contexto fornecido\n"
        "- Responda de forma completa e natural, como em uma conversa\n"
        "- Use APENAS informações presentes no contexto\n"
        "- Para perguntas sobre 'quando', forneça a resposta em uma frase completa mencionando o ano e o local\n"
        "- Seja preciso mas não robotizado\n"
        "- Evite respostas de uma única palavra ou número<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Contexto:\n{context}\n\n"
        "Pergunta: {question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    rag_chain = LLMChain(prompt=prompt, llm=llm)
    return rag_chain


def rerank_documents(question, retrieved_docs, cross_encoder, top_k):
    """Reordena os documentos usando um CrossEncoder."""
    pairs = [[question, doc.page_content] for doc in retrieved_docs]
    scores = cross_encoder.predict(pairs)
    doc_scores = list(zip(retrieved_docs, scores))
    doc_scores_sorted = sorted(doc_scores, key=lambda x: x[1], reverse=True)

    print(f"\n[DEBUG] Top {top_k} documentos após re-ranking:")
    for i, (doc, score) in enumerate(doc_scores_sorted[:top_k], 1):
        preview = doc.page_content[:80].replace('\n', ' ')
        print(f"  #{i} (score: {score:.4f}): {preview}...")

    return [doc for doc, score in doc_scores_sorted[:top_k]]


def ask(question, rag_chain, retriever, cross_encoder):
    """Pipeline completo: Retrieve -> Re-rank -> Generate."""
    print(f"\n{'='*80}")
    print(f"PERGUNTA: {question}")
    print('='*80)

    retrieved_docs = retriever.invoke(question)
    print(f"✓ Recuperados {len(retrieved_docs)} documentos")

    reranked_docs = rerank_documents(
        question, retrieved_docs, cross_encoder, TOP_K_AFTER_RERANK)

    context_parts = []
    for i, doc in enumerate(reranked_docs, 1):
        context_parts.append(f"[Trecho {i}]: {doc.page_content}")

    context = "\n\n".join(context_parts)

    print(f"\n[DEBUG] Primeiros 400 caracteres do contexto:")
    print("-" * 80)
    print(context[:400])
    print("-" * 80)

    result = rag_chain.invoke({"context": context, "question": question})
    answer = result['text'].strip()

    # Limpeza de tokens especiais do Llama 3.1
    tokens_to_remove = ["</s>", "<|eot_id|>",
                        "<|end_header_id|>", "<|start_header_id|>"]
    for token in tokens_to_remove:
        answer = answer.replace(token, "")

    answer = answer.strip()

    print(f"\n{'='*80}")
    print(f"RESPOSTA: {answer}")
    print('='*80)

    return answer, context


def main():
    try:
        print(f"\n{'='*80}")
        print(f"RAG com Llama 3.1 8B Instruct: {MODEL_BASENAME}")
        print('='*80)

        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_EMBEDDING_NAME,
            model_kwargs={'device': 'cuda'}
        )

        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        with open(os.path.join(DOCSTORE_PATH, "store.pkl"), "rb") as f:
            store = pickle.load(f)

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=50)
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter
        )
        retriever.search_kwargs['k'] = RETRIEVER_K

        print(f"\nCarregando cross-encoder...")
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

        print(f"\nBaixando modelo: {MODEL_BASENAME}...")
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID, filename=MODEL_BASENAME)
        print(f"✓ Modelo baixado: {model_path}")

        print("Inicializando LLM...")
        llm = load_local_llm(model_path)
        rag_chain = create_rag_chain(llm)

        evaluation_dataset = [
            # {
            #     "pergunta": "Quando o curso de Tecnologia em Análise e Desenvolvimento de Sistemas foi implantado no Campus de Picos?",
            #     "resposta_esperada": "O curso foi implantado no Campus de Picos em 2013."
            # },
            # {
            #     "pergunta": "Quais são as modalidades aceitas para a apresentação do TCC?",
            #     "resposta_esperada": "O TCC pode ser apresentado em formato de Monografia, Artigo Científico, Relatório Técnico de Software (RTS) ou Relatório Técnico de Trabalho/Estágio (RTT)."
            # },
            {
                "pergunta": "Qual foi o montante total de investimento que o setor de TI no Brasil atingiu em 2021, englobando os mercados de software, serviços, hardware e exportações?",
                "resposta_esperada": "O investimento atingiu R$238,2 bilhões (US$ 46,2 bilhões)."
            }
        ]

        results = []

        for item in evaluation_dataset:
            generated_answer, retrieved_context = ask(
                item["pergunta"],
                rag_chain,
                retriever,
                cross_encoder
            )

            results.append({
                "pergunta": item["pergunta"],
                "resposta_esperada": item["resposta_esperada"],
                "resposta_gerada": generated_answer,
                "contexto_recuperado": retrieved_context
            })

        df_results = pd.DataFrame(results)

        generated_answers = df_results["resposta_gerada"].tolist()
        expected_answers = df_results["resposta_esperada"].tolist()

        print("\n" + "="*80)
        print("CALCULANDO MÉTRICAS...")
        print("="*80)

        rouge_metric = evaluate.load('rouge')
        rouge_scores = rouge_metric.compute(
            predictions=generated_answers,
            references=expected_answers
        )

        bleu_metric = evaluate.load('bleu')
        bleu_scores = bleu_metric.compute(
            predictions=generated_answers,
            references=expected_answers
        )

        bertscore_metric = evaluate.load('bertscore')
        bertscore_scores = bertscore_metric.compute(
            predictions=generated_answers,
            references=expected_answers,
            lang='pt',
            model_type='distilbert-base-multilingual-cased'
        )

        def calculate_f1_token(prediction, reference):
            pred_tokens = set(prediction.lower().split())
            ref_tokens = set(reference.lower().split())

            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                return 0.0

            common = pred_tokens.intersection(ref_tokens)
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)

            if precision + recall == 0:
                return 0.0

            return 2 * (precision * recall) / (precision + recall)

        def calculate_exact_match(prediction, reference):
            return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0

        f1_scores = [calculate_f1_token(p, r) for p, r in zip(
            generated_answers, expected_answers)]
        em_scores = [calculate_exact_match(p, r) for p, r in zip(
            generated_answers, expected_answers)]

        avg_f1 = sum(f1_scores) / len(f1_scores)
        avg_em = sum(em_scores) / len(em_scores)
        avg_bertscore = sum(
            bertscore_scores['f1']) / len(bertscore_scores['f1'])

        print("\n" + "="*80)
        print("RESULTADOS FINAIS")
        print("="*80)
        print(f"Resposta Esperada: {expected_answers[0]}")
        print(f"Resposta Gerada  : {generated_answers[0]}")

        print("\n" + "="*80)
        print("MÉTRICAS DE AVALIAÇÃO")
        print("="*80)

        print("\n[ROUGE - Overlap de N-gramas]")
        print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
        print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
        print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")

        print("\n[BLEU - Precisão de N-gramas]")
        print(f"  BLEU Score: {bleu_scores['bleu']:.4f}")

        print("\n[BERTScore - Similaridade Semântica]")
        print(f"  F1 (média): {avg_bertscore:.4f}")
        print(
            f"  Precision:  {sum(bertscore_scores['precision'])/len(bertscore_scores['precision']):.4f}")
        print(
            f"  Recall:     {sum(bertscore_scores['recall'])/len(bertscore_scores['recall']):.4f}")

        print("\n[F1-Score Token-based]")
        print(f"  F1 (média): {avg_f1:.4f}")

        print("\n[Exact Match]")
        print(f"  Accuracy: {avg_em:.4f} ({int(avg_em*100)}%)")

        df_results.to_csv("resultados_rag_llama3.csv", index=False)
        print("\n✓ Resultados salvos em 'resultados_rag_llama3.csv'")

    finally:
        pass
        # os.kill(os.getpid(), 9)


if __name__ == "__main__":
    main()
