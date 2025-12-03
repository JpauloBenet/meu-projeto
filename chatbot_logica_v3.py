import os, pickle
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.chains.llm import LLMChain
from langchain_cohere import CohereRerank
from configs_v2 import get_config 

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
K_INITIAL_SEARCH = 30
RERANKER_TOP_N = 18
MAX_DOCS_TO_REFINE = 3
K_REFINED_SEARCH = 25

REWRITE_PROMPT_TEMPLATE = """
Você é um engenheiro de busca sênior, especialista em otimizar perguntas de usuários para um sistema de busca em documentos regulatórios do Banco Central.
Sua tarefa é reescrever a pergunta do usuário, transformando-a em uma única query de busca vetorial, precisa e autocontida.

**Siga estas diretrizes para a reescrita:**
1.  **Extraia o Intento Principal:** Identifique o objetivo central da pergunta do usuário.
2.  **Incorpore Termos-Chave:** Isole e inclua todos os termos técnicos, entidades, números de normas ou artigos mencionados (ex: "Capital Principal", "Resolução 4.958", "Art. 66", "FPR").
3.  **Desambigue e Expanda:** Se houver acrônimos ou termos ambíguos, adicione contexto ou o nome por extenso para tornar a busca mais precisa (ex: "CRI" -> "Certificado de Recebíveis Imobiliários").
4.  **Formule uma Pergunta Clara:** Construa uma pergunta completa e direta, como se estivesse consultando um especialista na norma. Remova qualquer informalidade ou texto supérfluo.

**Não responda à pergunta.** Apenas forneça a versão otimizada para a busca.

Pergunta Original: "{question}"
Pergunta Otimizada para Busca:"""

def obter_resposta_v3(question: str, llm: ChatOpenAI, vectorstore: FAISS, bm25_retriever_full: BM25Retriever) -> Dict[str, Any]:
    print("--- INICIANDO MOTOR 'Otimizado 3.0 - MODO DIAGNÓSTICO' ---")

    rewrite_prompt = PromptTemplate.from_template(REWRITE_PROMPT_TEMPLATE)
    rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt)
    rewritten_question = rewrite_chain.invoke({"question": question})['text']

    print("\n" + "="*50)
    print("--- PERGUNTA REESCRITA GERADA ---")
    print(rewritten_question)
    print("="*50 + "\n")


    faiss_retriever_initial = vectorstore.as_retriever(search_kwargs={"k": K_INITIAL_SEARCH})
    bm25_retriever_full.k = K_INITIAL_SEARCH
    ensemble_retriever_initial = EnsembleRetriever(retrievers=[bm25_retriever_full, faiss_retriever_initial], weights=[0.5, 0.5])
    
    initial_chunks = ensemble_retriever_initial.get_relevant_documents(rewritten_question)
    
    if not initial_chunks:
        return {"answer": "Com base nos documentos fornecidos, não encontrei informações para responder a essa pergunta.", "source_documents": []}

    reranker = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-multilingual-v3.0", top_n=RERANKER_TOP_N)
    reranked_chunks = reranker.compress_documents(documents=initial_chunks, query=question)
    if not reranked_chunks:
        return {"answer": "Com base nos documentos fornecidos, não encontrei informações para responder a essa pergunta.", "source_documents": []}

    source_documents = [chunk.metadata.get('origem') for chunk in reranked_chunks if chunk.metadata.get('origem')]
    unique_sources = list(dict.fromkeys(source_documents))
    docs_to_refine = unique_sources[:MAX_DOCS_TO_REFINE]

    refined_chunks_all = []
    for doc_origin in docs_to_refine:
        faiss_retriever_filtered = vectorstore.as_retriever(search_kwargs={'k': K_REFINED_SEARCH, 'filter': {'origem': doc_origin}})
        docs_for_bm25_filtered = [doc for doc in bm25_retriever_full.docs if doc.metadata.get('origem') == doc_origin]
        if not docs_for_bm25_filtered: continue
        
        bm25_retriever_filtered = BM25Retriever.from_documents(docs_for_bm25_filtered)
        bm25_retriever_filtered.k = K_REFINED_SEARCH
        ensemble_retriever_filtered = EnsembleRetriever(retrievers=[bm25_retriever_filtered, faiss_retriever_filtered], weights=[0.5, 0.5])
        
        chunks_from_this_doc = ensemble_retriever_filtered.get_relevant_documents(rewritten_question)
        refined_chunks_all.extend(chunks_from_this_doc)

    final_context_docs = reranked_chunks if not refined_chunks_all else list({doc.page_content: doc for doc in refined_chunks_all}.values())
    context_text = "\n\n---\n\n".join([doc.page_content for doc in final_context_docs])
    print("\n" + "="*50)
    print("--- CONTEXTO FINAL ENVIADO AO LLM ---")
    print(context_text)
    print("="*50 + "\n")

    qa_prompt = PromptTemplate(template=get_config('prompt'), input_variables=["context", "question"])
    final_chain = LLMChain(llm=llm, prompt=qa_prompt)
    response = final_chain.invoke({"context": context_text, "question": question})
    print("--- FINALIZANDO MOTOR 'Otimizado 3.0 - MODO DIAGNÓSTICO' ---")
    return {"answer": response['text'], "source_documents": final_context_docs}