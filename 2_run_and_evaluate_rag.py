import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from huggingface_hub import hf_hub_download
import evaluate  # Biblioteca da Hugging Face para métricas

# --- CONFIGURAÇÕES ---
INDEX_PATH = "faiss_index"
MODEL_EMBEDDING_NAME = "BAAI/bge-m3"


def load_local_llm(model_path):
    """Carrega o LLM localmente usando LlamaCpp com aceleração de GPU."""
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Parâmetros para sua RTX 3060 (6GB VRAM)
    # n_gpu_layers: Número de camadas a serem descarregadas para a GPU.
    # Um valor entre 20 e 30 deve funcionar bem para o Mistral 7B Q5_K_M.
    n_gpu_layers = 25

    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=512,
        n_ctx=4096,  # Tamanho do contexto
        f16_kv=True,  # Deve ser True para performance
        callback_manager=callback_manager,
        verbose=True,
        temperature=0.1,  # Temperatura baixa para respostas mais factuais
        max_tokens=1024
    )
    return llm


def create_rag_chain(llm, retriever):
    """Cria a cadeia RAG com um prompt template otimizado."""
    template = """
    Use estritamente as informações do contexto abaixo para responder à pergunta.
    Se a resposta não estiver contida no contexto, diga "Não encontrei a resposta no contexto fornecido".
    Não invente informações. Seja conciso e direto.

    Contexto: {context}

    Pergunta: {question}

    Resposta:
    """
    prompt = PromptTemplate(template=template, input_variables=[
                            "context", "question"])
    rag_chain = LLMChain(prompt=prompt, llm=llm)
    return rag_chain


def ask(question, rag_chain, retriever):
    """Faz uma pergunta ao sistema RAG e retorna a resposta e o contexto recuperado."""
    print(f"\n--- Processando Pergunta: {question} ---")
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    result = rag_chain.invoke({"context": context, "question": question})
    return result['text'], context


def main():
    # Carrega o modelo de embedding
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_EMBEDDING_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Carrega o índice FAISS
    vectorstore = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    # Recupera os 4 chunks mais relevantes
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

    # Baixa e carrega o LLM
    model_name_gguf = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    model_basename = "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
    model_path = hf_hub_download(
        repo_id=model_name_gguf, filename=model_basename)
    llm = load_local_llm(model_path)

    # Cria a cadeia RAG
    rag_chain = create_rag_chain(llm, retriever)


    # --- AVALIAÇÃO COM GOLDEN DATASET ---
    # Crie seu dataset de avaliação. Quanto mais itens, mais confiáveis as métricas.
    evaluation_dataset = [
        {
            "pergunta": "Qual é o objetivo geral do Curso Superior de Tecnologia em Análise e Desenvolvimento de Sistemas?",
            "resposta_esperada": "O objetivo geral é formar profissionais para o mercado de TI com competências para propor, analisar, projetar, desenvolver, implementar e atualizar sistemas de informação, programar computadores e desenvolver softwares."
        },
        {
            "pergunta": "Qual a carga horária total do curso de ADS do IFPI Campus Picos e como ela é distribuída?",
            "resposta_esperada": "A carga horária total do curso é de 2100 horas, distribuídas em 2000 horas de disciplinas obrigatórias e 100 horas de atividades complementares."
        },
        {
            "pergunta": "O estágio é obrigatório para o curso de Tecnologia em Análise e Desenvolvimento de Sistemas?",
            "resposta_esperada": "Não, o estágio não é obrigatório no Curso de Tecnologia em Análise e Desenvolvimento de Sistemas, sendo desenvolvido como uma atividade opcional."
        },
        {
            "pergunta": "Quais são as formas de ingresso para os cursos superiores de graduação do IFPI?",
            "resposta_esperada": "O ingresso acontece mediante processo seletivo público, como Vestibular, Exame Nacional do Ensino Médio (ENEM), transferências e portadores de diplomas."
        },
        {
            "pergunta": "Quantas vagas anuais são ofertadas para o curso de ADS no campus Picos e qual o turno de funcionamento?",
            "resposta_esperada": "São ofertadas 40 vagas por ano para o turno vespertino."
        },
        {
            "pergunta": "Qual o pré-requisito para cursar a disciplina de Inteligência Artificial?",
            "resposta_esperada": "O pré-requisito para a disciplina de Inteligência Artificial é ter cursado a disciplina de Estatística, código 07."
        },
        {
            "pergunta": "Quais são as modalidades de Trabalho de Conclusão de Curso (TCC) aceitas?",
            "resposta_esperada": "O TCC pode ser apresentado em formato de Monografia, Artigo Científico, Relatório Técnico de Software (RTS) ou Relatório Técnico de Trabalho/Estágio (RTT)."
        },
        {
            "pergunta": "Cite duas disciplinas optativas que os alunos de ADS podem cursar.",
            "resposta_esperada": "Os alunos podem cursar disciplinas como Algoritmos de Aprendizado de Máquina, Processamento de Línguas Naturais, Processamento de Imagens, Introdução a Ciência de Dados, Ética e Responsabilidade Socioambiental ou Libras."
        },
        {
            "pergunta": "Qual a média final mínima para aprovação após o exame final?",
            "resposta_esperada": "A média final para aprovação após o exame final deve ser igual ou superior a 6,0 (seis)."
        },
        {
            "pergunta": "Quando o curso de Tecnologia em Análise e Desenvolvimento de Sistemas foi implantado no Campus de Picos?",
            "resposta_esperada": "O curso foi implantado no Campus de Picos em 2013."
        }
    ]

    results = []
    print("\n--- INICIANDO AVALIAÇÃO ---")
    for item in evaluation_dataset:
        generated_answer, retrieved_context = ask(
            item["pergunta"], rag_chain, retriever)
        results.append({
            "pergunta": item["pergunta"],
            "resposta_esperada": item["resposta_esperada"],
            "resposta_gerada": generated_answer,
            "contexto_recuperado": retrieved_context
        })

    df_results = pd.DataFrame(results)

    # --- CÁLCULO DAS MÉTRICAS ---
    rouge_metric = evaluate.load('rouge')

    generated_answers = df_results["resposta_gerada"].tolist()
    expected_answers = df_results["resposta_esperada"].tolist()

    rouge_scores = rouge_metric.compute(
        predictions=generated_answers, references=expected_answers)

    print("\n--- RESULTADOS DA AVALIAÇÃO ---")
    print("Métricas ROUGE:")
    print(rouge_scores)

    # Salvando os resultados para análise detalhada
    df_results.to_csv("resultados_rag.csv", index=False)
    print("\nResultados completos salvos em 'resultados_rag.csv'")


if __name__ == "__main__":
    main()
