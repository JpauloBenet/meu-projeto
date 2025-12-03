import os
from typing import Dict, Any, List
from langchain_core.documents import Document
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.llm import LLMChain

from motor_unificado import obter_resposta_unificada

load_dotenv()

# --- DEFINIÇÃO DOS PROMPTS PARA O ROTEADOR ---

ROUTER_PROMPT_TEMPLATE = """
Sua tarefa é classificar a pergunta do usuário em uma de duas categorias, com base em sua intenção. Responda apenas com o nome da categoria.

Categorias:
- "Consulta Normativa": A pergunta envolve a interpretação, cálculo, aplicação ou detalhes de uma lei, regra, norma, resolução ou artigo.
- "Consulta Factual": A pergunta busca uma definição, descrição ou informação geral sobre uma entidade, pessoa, produto ou conceito.

Pergunta do Usuário:
"{question}"

Categoria:
"""

FACTUAL_PROMPT_TEMPLATE = """
Você é um assistente de conhecimento, especialista em extrair e apresentar informações de forma clara e organizada. Sua tarefa é responder à pergunta do usuário criando um resumo completo e bem estruturado, utilizando apenas as informações fornecidas no contexto abaixo.

**Instruções:**
1.  **Síntese Completa:** Leia todo o contexto e sintetize as informações para fornecer uma resposta abrangente.
2.  **Estrutura Lógica:** Organize a resposta em parágrafos. Comece com uma definição ou resumo geral e, em seguida, adicione detalhes importantes, como datas, classificações, produtos ou objetivos.
3.  **Linguagem Clara:** Use uma linguagem direta e clara. Evite jargões, a menos que sejam explicados.
4.  **Fidelidade ao Contexto:** Não adicione nenhuma informação que não esteja explicitamente no texto do contexto. Se a informação não estiver lá, não a invente.

Contexto:
{context}

Pergunta do Usuário:
{question}

Resposta Completa e Estruturada:
"""


# --- FUNÇÃO PRINCIPAL DO MOTOR ROTEADOR (MODIFICADA) ---

def obter_resposta_roteada(
    question: str,
    chat_history: List[Dict[str, str]],
    llm: ChatOpenAI,
    vectorstore: Any,
    bm25_retriever_full: Any,
    ordered_chunks: List[Document],
    available_norms: List[str]
) -> Dict[str, Any]:
    """
    Primeiro, classifica a intenção da pergunta e depois a direciona
    para o pipeline de RAG apropriado.
    """
    print("--- INICIANDO MOTOR ROTEADOR ---")

    # ETAPA 1: CLASSIFICAÇÃO DA INTENÇÃO
    print(f"-> Classificando a pergunta: '{question}'")
    router_prompt = PromptTemplate.from_template(ROUTER_PROMPT_TEMPLATE)
    router_chain = router_prompt | llm | StrOutputParser()
    intent = router_chain.invoke({"question": question}).strip()
    print(f"--> Intenção detectada: {intent}")

    # ETAPA 2: ROTEAMENTO PARA O ESPECIALISTA CORRETO

    if intent == "Consulta Normativa":
        print("--> Roteando para o Motor Unificado.")
        return obter_resposta_unificada(
            question=question,
            chat_history=chat_history,
            llm=llm,
            vectorstore=vectorstore,
            bm25_retriever_full=bm25_retriever_full,
            ordered_chunks=ordered_chunks,
            available_norms=available_norms
        )
    
    elif intent == "Consulta Factual":
        print("--> Roteando para o Motor Factual Simples.")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        docs = retriever.invoke(question)
        
        if not docs:
            return {"answer": "Desculpe, não encontrei informações sobre este tópico.", "source_documents": []}
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        factual_prompt = PromptTemplate.from_template(FACTUAL_PROMPT_TEMPLATE)
        factual_chain = LLMChain(llm=llm, prompt=factual_prompt)
        response = factual_chain.invoke({"context": context_text, "question": question})
        
        return {"answer": response['text'], "source_documents": docs}

    else:
        print("--> Não foi possível classificar a intenção, usando o motor padrão (Unificado).")
        return obter_resposta_unificada(
            question=question,
            chat_history=chat_history,
            llm=llm,
            vectorstore=vectorstore,
            bm25_retriever_full=bm25_retriever_full,
            ordered_chunks=ordered_chunks,
            available_norms=available_norms
        )