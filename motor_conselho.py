import os
import re
import concurrent.futures
from typing import Dict, Any, List
from langchain_core.documents import Document
from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Importa as funções de resposta dos outros motores.
from chatbot_logica_v2 import obter_resposta_v2
from chatbot_logica_v3 import obter_resposta_v3
from chatbot_logica_v4 import obter_resposta_v4
from chatbot_logica_v5 import obter_resposta_v5

# Importa o template de formatação final.
from configs_v2 import FINAL_FORMATTER_TEMPLATE

load_dotenv()

# --- DEFINIÇÃO DOS PROMPTS COMPLETOS ---

CONDENSE_QUESTION_PROMPT = """
Dado o histórico de uma conversa e uma nova pergunta, reformule a nova pergunta para ser uma pergunta autônoma e completa, em português, que possa ser entendida sem o histórico. NÃO responda à pergunta, apenas a reformule.

Histórico da Conversa:
{chat_history}

Nova Pergunta:
{question}

Pergunta Autônoma:
"""

SYNTHESIS_PROMPT_TEMPLATE = """
Você é um analista regulatório sênior do Banco Central do Brasil, encarregado de produzir uma análise definitiva e completa. Sua tarefa é ler todo o CONTEXTO NORMATIVO fornecido, que contém trechos de diferentes resoluções, e sintetizar uma única resposta coesa e bem fundamentada para a PERGUNTA DO USUÁRIO.

**Instrução Mestra de Síntese:**
1.  **Leia e Conecte:** Analise todo o contexto. Identifique como as diferentes normas se complementam. Uma norma pode fornecer a fórmula principal, enquanto outra detalha a metodologia de cálculo de seus componentes. Sua principal tarefa é conectar esses pontos.
2.  **Estruture a Resposta:** Siga rigorosamente a estrutura de resposta padrão (Análise dos Fatos, Dedução Lógica, Conclusão).
3.  **Fundamente nos Fatos:** Baseie cada afirmação exclusivamente nos trechos fornecidos no CONTEXTO NORMATIVO. Cite os artigos e normas relevantes.
4.  **Seja Conclusivo:** Não se refira a "outros analistas" ou "pareceres". A análise é sua. Você é a autoridade final.

**Regra de Segurança:**
Se, mesmo após analisar todo o contexto, as informações forem insuficientes para responder à pergunta, declare isso claramente.

---
**CONTEXTO NORMATIVO (Trechos de todas as normas relevantes encontradas):**
{consolidated_context}

---
**PERGUNTA DO USUÁRIO:**
{question}

---
**Sua Análise Final (siga a estrutura padrão):**
"""


# --- FUNÇÕES AUXILIARES ---

def parse_query_for_metadata(question: str) -> Dict[str, str]:
    """Analisa a pergunta do usuário para extrair menções explícitas a normas."""
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

def extract_metadata_from_filename(file: Path) -> Dict[str, str]:
    """Extrai metadados do nome de um arquivo para encontrar a norma correspondente."""
    name = file.stem
    name_lower = name.lower()
    meta = {}
 
    patterns = {
        "resolucao": r"^(?:res_|resolucao_?)(\d+)",
        "carta circular": r"^(?:carta_circular_|c[_\s]?circ[_\s]?|circ[_\s]?)(\d+)",
        "circular": r"^(?:circular_|circ[_\s]?)(\d+)",
        "instrucao": r"^(?:dlo_|instrucao_?)(\d+)",
        "norma": r"^norma[_\s]?(\d+)",
        "instrumento": r"^(?:instrumento_|intrumento_?)(\d+)",
        "voto": r"^(?:Voto_|VOTO_?)(\d+)",
        "contexto": r"^(?:contexto[_\s]?)(\d+)",
    }
    for norma_type, pattern in patterns.items():
        m = re.match(pattern, name_lower)
        if m:
            norma_type_normalized = 'resolucao' if 'resolucao' in norma_type else norma_type
            norma_type_normalized = 'circular' if 'circular' in norma_type and 'carta' not in norma_type else norma_type_normalized
            return {"tipo_norma": norma_type_normalized, "numero_norma": m.group(1)}
    return {}


# --- FUNÇÃO PRINCIPAL DO MOTOR DE SÍNTESE ---

def obter_resposta_conselho(
    question: str,
    chat_history: List[Dict[str, str]],
    llm: ChatOpenAI,
    vectorstore: Any,
    bm25_retriever_full: Any,
    ordered_chunks: List[Document],
    available_norms: List[str]  
) -> Dict[str, Any]:
    """
    Executa um pipeline de RAG em múltiplas etapas:
    1. Expansão: Coleta documentos de vários motores de busca em paralelo.
    2. Síntese: Usa um LLM avançado para gerar uma resposta a partir do contexto consolidado.
    """
    print("--- INICIANDO MOTOR DE 'SÍNTESE AVANÇADA' (RAG Multi-Etapas) ---")

    # ETAPA 0: CONDENSAÇÃO DA PERGUNTA COM HISTÓRICO
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
    print(f"-> Pergunta autônoma gerada: {standalone_question}")


    # --- ETAPA 1: EXPANSÃO (Coleta de Documentos) ---
    print("--- ETAPA 1: EXPANSÃO (Coletando documentos de todos os motores) ---")
    
    norma_mencionada_meta = parse_query_for_metadata(standalone_question)
    normas_focadas = []
    if norma_mencionada_meta:
        for norm_filename_str in available_norms:
            file_meta = extract_metadata_from_filename(Path(norm_filename_str))
            if file_meta == norma_mencionada_meta:
                normas_focadas.append(norm_filename_str)
                break
    
    all_source_docs = []
    base_kwargs = {"question": standalone_question, "llm": llm, "vectorstore": vectorstore, "bm25_retriever_full": bm25_retriever_full}
    kwargs_com_widening = {**base_kwargs, "ordered_chunks": ordered_chunks}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(obter_resposta_v2, **kwargs_com_widening),
            executor.submit(obter_resposta_v3, **base_kwargs),
            executor.submit(obter_resposta_v4, **kwargs_com_widening)
        ]
        if normas_focadas:
            print(f"--> Adicionando busca focada (v5) na norma '{normas_focadas[0]}' à etapa de expansão.")
            kwargs_v5 = {**kwargs_com_widening, "normas_selecionadas": normas_focadas}
            futures.append(executor.submit(obter_resposta_v5, **kwargs_v5))

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                all_source_docs.extend(result.get('source_documents', []))
            except Exception as e:
                print(f"Erro ao obter resultado de um motor: {e}")

    unique_docs_dict = {doc.page_content: doc for doc in all_source_docs}
    unique_source_docs = list(unique_docs_dict.values())
    
    if not unique_source_docs:
        return {"answer": "Não foi possível encontrar nenhum documento relevante para responder à pergunta.", "source_documents": []}

    print(f"-> Expansão concluída. {len(unique_source_docs)} trechos de documentos únicos foram encontrados.")


    # --- ETAPA 2: SÍNTESE (Geração da Resposta Final) ---
    print("--- ETAPA 2: SÍNTESE (Gerando resposta a partir do contexto consolidado) ---")

    consolidated_context = "\n\n---\n\n".join([doc.page_content for doc in unique_source_docs])

    synthesis_prompt = PromptTemplate(template=SYNTHESIS_PROMPT_TEMPLATE, input_variables=["consolidated_context", "question"])
    synthesis_llm = ChatCohere(model="command-r-plus-08-2024", temperature=0, cohere_api_key=os.getenv("COHERE_API_KEY"))
    synthesis_chain = synthesis_prompt | synthesis_llm | StrOutputParser()
    
    internal_verdict = synthesis_chain.invoke({
        "consolidated_context": consolidated_context,
        "question": standalone_question
    })

    print("-> Formatando a análise final...")
    formatter_prompt = PromptTemplate.from_template(FINAL_FORMATTER_TEMPLATE)
    formatting_chain = formatter_prompt | llm | StrOutputParser()
    final_answer = formatting_chain.invoke({
        "verified_analysis": internal_verdict,
        "question": standalone_question
    })

    print("--- FINALIZANDO MOTOR DE 'SÍNTESE AVANÇADA' ---")
    return {"answer": final_answer, "source_documents": unique_source_docs}
