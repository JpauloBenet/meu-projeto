
import os, pickle, re
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.llm import LLMChain
from langchain_cohere import CohereRerank
from configs_v2 import get_config

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
K_INITIAL_SEARCH = 30
RERANKER_TOP_N = 10
HYDE_TEMPLATE = """
Você é um especialista em regulação do BACEN.
Escreva um parágrafo que responda de forma plausível à seguinte pergunta.
O parágrafo deve ser um trecho de um documento normativo fictício, usando linguagem técnica e formal.

Pergunta: {question}
Resposta Fictícia:
"""

def parse_query_for_metadata(question: str) -> Dict[str, str]:
    question_lower = question.lower().strip()
    patterns = {
        'carta circular': r'(?:carta circular|c_circ|circ)\s*n?º?\s*(\d+)',
        'circular': r'circular\s*n?º?\s*(\d+)(?!.*carta)',
        'resolucao': r'(?:resolucao|res)\s*n?º?\s*(\d+)',
    }
    for norma_type, pattern in patterns.items():
        match = re.search(pattern, question_lower)
        if match:
            return {"tipo_norma": norma_type, "numero_norma": match.group(1)}
    return {}

def get_context_from_metadata_filter(vectorstore: FAISS, metadata_filter: Dict) -> List[Document]:
    print(f"-> Executando busca direta por filtro: {metadata_filter}")
    relevant_docs = []
    for doc_id in vectorstore.index_to_docstore_id.values():
        doc = vectorstore.docstore.search(doc_id)
        if (doc is not None and
            doc.metadata.get('tipo_norma') == metadata_filter['tipo_norma'] and
            doc.metadata.get('numero_norma') == metadata_filter['numero_norma']):
            relevant_docs.append(doc)
    return relevant_docs

def run_full_rag_pipeline(question: str, llm: ChatOpenAI, vectorstore: FAISS, bm25_retriever_full: BM25Retriever, ordered_chunks: List[Document]) -> List[Document]:
    print("--- Executando Pipeline Semântico Completo (HyDE) ---")
    hyde_prompt = PromptTemplate(template=HYDE_TEMPLATE, input_variables=["question"])
    hyde_chain = hyde_prompt | llm | StrOutputParser()
    hypothetical_document = hyde_chain.invoke({"question": question})

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": K_INITIAL_SEARCH})
    bm25_retriever_full.k = K_INITIAL_SEARCH
    
    initial_chunks_faiss = faiss_retriever.invoke(hypothetical_document)
    initial_chunks_bm25 = bm25_retriever_full.invoke(question)
    initial_chunks = list({doc.page_content: doc for doc in initial_chunks_faiss + initial_chunks_bm25}.values())

    if not initial_chunks: return []

    reranker = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-multilingual-v3.0", top_n=RERANKER_TOP_N)
    reranked_chunks = reranker.compress_documents(documents=initial_chunks, query=question)
    if not reranked_chunks: return []

    final_docs_dict = {}
    if not reranked_chunks[0].metadata.get("origem"): return reranked_chunks
    
    primary_source_document = reranked_chunks[0].metadata["origem"]
    for chunk in reranked_chunks:
        final_docs_dict[chunk.page_content] = chunk
        if chunk.metadata.get("origem") == primary_source_document:
            chunk_index = chunk.metadata.get('original_index')
            if chunk_index is not None:
                start_index = max(0, chunk_index - 1)
                end_index = min(len(ordered_chunks), chunk_index + 2)
                for neighbor in ordered_chunks[start_index:end_index]:
                    final_docs_dict[neighbor.page_content] = neighbor
    return list(final_docs_dict.values())

def obter_resposta_v4(question: str, llm: ChatOpenAI, vectorstore: FAISS, bm25_retriever_full: BM25Retriever, ordered_chunks: List[Document]) -> Dict[str, Any]:
    print("--- INICIANDO MOTOR 'Híbrido v2.0 (HyDE + Foco)' ---")
    final_context_docs = []
    metadata_filter = parse_query_for_metadata(question)

    if metadata_filter:
        final_context_docs = get_context_from_metadata_filter(vectorstore, metadata_filter)

    if not final_context_docs:
        final_context_docs = run_full_rag_pipeline(question, llm, vectorstore, bm25_retriever_full, ordered_chunks)

    if not final_context_docs:
        return {"answer": "Com base nos documentos fornecidos, não encontrei informações para responder a essa pergunta.", "source_documents": []}

    context_text = "\n\n---\n\n".join([doc.page_content for doc in final_context_docs])
    qa_prompt = PromptTemplate(template=get_config('prompt'), input_variables=["context", "question"])
    final_chain = LLMChain(llm=llm, prompt=qa_prompt)
    response = final_chain.invoke({"context": context_text, "question": question})
    print("--- FINALIZANDO MOTOR 'Híbrido v2.0 (HyDE + Foco)' ---")
    return {"answer": response['text'], "source_documents": final_context_docs}