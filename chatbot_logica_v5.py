import os
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.chains.llm import LLMChain
from langchain_cohere import CohereRerank
from configs_v2 import get_config

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
RERANKER_TOP_N = 10
TOP_N_FOR_WIDENING = 5
K_SEARCH_PER_NORM = 20

def obter_resposta_v5(
    question: str, 
    llm: ChatOpenAI, 
    vectorstore: FAISS, 
    bm25_retriever_full: BM25Retriever, 
    ordered_chunks: List[Dict[str, Any]],
    normas_selecionadas: List[str]
) -> Dict[str, Any]:
    """
    Motor de RAG com foco específico, que realiza a busca apenas nas normas
    selecionadas manualmente pelo usuário, usando filtragem por metadados.
    Baseado na robusta pipeline v2 (Ensemble + Rerank + Widening).
    """
    print(f"--- INICIANDO MOTOR 'Foco Específico v1.0' ---")
    print(f"Normas selecionadas para a busca: {normas_selecionadas}")

    if not normas_selecionadas:
        return {"answer": "Por favor, selecione ao menos uma norma para realizar a busca focada.", "source_documents": []}

    docs_for_bm25_filtered = [
        doc for doc in bm25_retriever_full.docs 
        if doc.metadata.get('origem') in normas_selecionadas
    ]
    if not docs_for_bm25_filtered:
        print("AVISO: Nenhum documento encontrado para as normas selecionadas no índice BM25.")
        bm25_chunks = []
    else:
        bm25_retriever_filtered = BM25Retriever.from_documents(docs_for_bm25_filtered)
        bm25_retriever_filtered.k = K_SEARCH_PER_NORM
        bm25_chunks = bm25_retriever_filtered.invoke(question)

    all_faiss_chunks = []
    for norma in normas_selecionadas:
        retriever_for_norm = vectorstore.as_retriever(
            search_kwargs={'k': K_SEARCH_PER_NORM, 'filter': {'origem': norma}}
        )
        chunks_from_norm = retriever_for_norm.invoke(question)
        all_faiss_chunks.extend(chunks_from_norm)

    initial_chunks_dict = {}
    for chunk in bm25_chunks + all_faiss_chunks:
        initial_chunks_dict[chunk.page_content] = chunk
    initial_chunks = list(initial_chunks_dict.values())

    if not initial_chunks:
        return {"answer": "Não encontrei trechos relevantes nas normas selecionadas para responder a essa pergunta.", "source_documents": []}


    reranker = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-multilingual-v3.0", top_n=RERANKER_TOP_N)
    reranked_chunks = reranker.compress_documents(documents=initial_chunks, query=question)
    
    if not reranked_chunks:
        return {"answer": "Após o re-ranking, nenhum trecho foi considerado relevante para a pergunta.", "source_documents": []}

    final_docs_dict = {}
    for relevant_chunk in reranked_chunks[:TOP_N_FOR_WIDENING]:
        final_docs_dict[relevant_chunk.page_content] = relevant_chunk
        chunk_index = relevant_chunk.metadata.get('original_index')
        if chunk_index is not None:
            start_index = max(0, chunk_index - 1)
            end_index = min(len(ordered_chunks), chunk_index + 2)
            neighbor_chunks = ordered_chunks[start_index:end_index]
            for chunk in neighbor_chunks:
                if chunk.metadata.get('origem') in normas_selecionadas:
                    final_docs_dict[chunk.page_content] = chunk
    
    final_context_docs = list(final_docs_dict.values())
    context_text = "\n\n---\n\n".join([doc.page_content for doc in final_context_docs])

    qa_prompt = PromptTemplate(template=get_config('prompt'), input_variables=["context", "question"])
    final_chain = LLMChain(llm=llm, prompt=qa_prompt)
    response = final_chain.invoke({"context": context_text, "question": question})

    print("--- FINALIZANDO MOTOR 'Foco Específico v1.0' ---")
    return {"answer": response['text'], "source_documents": final_context_docs}