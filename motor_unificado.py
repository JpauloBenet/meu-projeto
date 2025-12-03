import os
from typing import List, Dict, Any

from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain.chains.llm import LLMChain
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser

from configs_v2 import FINAL_FORMATTER_TEMPLATE
from extrator_metadados import ExtratorDeMetadados
from motor_conselho import CONDENSE_QUESTION_PROMPT, SYNTHESIS_PROMPT_TEMPLATE

K_INITIAL_SEARCH = 70
RERANKER_TOP_N = 9

def obter_resposta_unificada(
    question: str,
    chat_history: List[Dict[str, str]],
    llm: ChatOpenAI,
    vectorstore: FAISS,
    bm25_retriever_full: BM25Retriever,
    ordered_chunks: List[Document],
    available_norms: List[str] 
) -> Dict[str, Any]:
    """
    Executa um pipeline de RAG unificado e sequencial:
    1. Condensa a pergunta com base no histórico.
    2. Extrai metadados da pergunta para criar filtros de busca.
    3. Executa uma busca híbrida (BM25 + Vetorial) usando os filtros.
    4. Re-rankeia os resultados para obter os melhores candidatos.
    5. Usa uma cadeia de LLMs (Síntese + Formatação) para gerar a resposta final.
    """
    print("\n--- INICIANDO MOTOR UNIFICADO v1.0 (Sequencial) ---")

    # ETAPA 1: CONDENSAÇÃO DA PERGUNTA (Lógica reaproveitada do motor_conselho)
    formatted_chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    if not formatted_chat_history:
        standalone_question = question
    else:
        condense_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT)
        condense_chain = condense_prompt | llm | StrOutputParser()
        standalone_question = condense_chain.invoke({
            "chat_history": formatted_chat_history,
            "question": question
        })
    print(f"-> Pergunta autônoma: {standalone_question}")

    # ETAPA 2: EXTRAÇÃO DE METADADOS PARA FILTRAGEM
    metadata_extractor = ExtratorDeMetadados(llm)
    extracted_filters = metadata_extractor.extrair_filtros(standalone_question)

    # ETAPA 3: RECUPERAÇÃO HÍBRIDA E FILTRADA
    print("--- ETAPA 3: Recuperação Híbrida Filtrada ---")

    search_kwargs = {'k': K_INITIAL_SEARCH}
    if extracted_filters:

        faiss_filter = {k: v for k, v in extracted_filters.items() if k in ["tipo_norma", "numero_norma"]}
        if faiss_filter:
             search_kwargs['filter'] = faiss_filter

    faiss_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    bm25_retriever_full.k = K_INITIAL_SEARCH

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever_full, faiss_retriever], weights=[0.5, 0.5]
    )
    
    initial_chunks = ensemble_retriever.invoke(standalone_question)
    print(f"-> {len(initial_chunks)} documentos recuperados na busca inicial.")
    if not initial_chunks:
        return {"answer": "Com base nos documentos fornecidos, não encontrei informações para responder a essa pergunta.", "source_documents": []}

    # ETAPA 4: RE-RANKING (Lógica reaproveitada)
    print("--- ETAPA 4: Re-ranking dos documentos ---")
    reranker = CohereRerank(cohere_api_key=os.getenv("COHERE_API_KEY"), model="rerank-multilingual-v3.0", top_n=RERANKER_TOP_N)
    reranked_chunks = reranker.compress_documents(documents=initial_chunks, query=standalone_question)
    
    unique_docs_dict = {doc.page_content: doc for doc in reranked_chunks}
    final_context_docs = list(unique_docs_dict.values())

    if not final_context_docs:
        return {"answer": "Após o re-ranking, nenhum trecho foi considerado relevante para a pergunta.", "source_documents": []}
    print(f"-> {len(final_context_docs)} documentos unicos após re-ranking.")


    # ETAPA 5: SÍNTESE E FORMATAÇÃO (Lógica reaproveitada do motor_conselho)
    print("--- ETAPA 5: Geração da Resposta Final (Síntese + Formatação) ---")
    consolidated_context = "\n\n---\n\n".join([doc.page_content for doc in final_context_docs])

    synthesis_prompt = PromptTemplate(template=SYNTHESIS_PROMPT_TEMPLATE, input_variables=["consolidated_context", "question"])
    synthesis_llm = ChatCohere(model="command-r-plus-08-2024", temperature=0, cohere_api_key=os.getenv("COHERE_API_KEY"))
    synthesis_chain = synthesis_prompt | synthesis_llm | StrOutputParser()
    internal_verdict = synthesis_chain.invoke({
        "consolidated_context": consolidated_context,
        "question": standalone_question
    })

    formatter_prompt = PromptTemplate.from_template(FINAL_FORMATTER_TEMPLATE)
    formatting_chain = formatter_prompt | llm | StrOutputParser()
    final_answer = formatting_chain.invoke({
        "verified_analysis": internal_verdict,
        "question": standalone_question
    })
    
    print("--- FINALIZANDO MOTOR UNIFICADO ---")
    return {"answer": final_answer, "source_documents": final_context_docs}