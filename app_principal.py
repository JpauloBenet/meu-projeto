import os
import json
import pickle
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List

# Importações dos motores
from chatbot_logica_v2 import obter_resposta_v2
from chatbot_logica_v4 import obter_resposta_v4
from chatbot_logica_v3 import obter_resposta_v3
from chatbot_logica_v5 import obter_resposta_v5 
from motor_conselho import obter_resposta_conselho
from motor_unificado import obter_resposta_unificada # <-- NOVA IMPORTAÇÃO
from configs_v2 import get_config

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# O caminho da pasta de dados pode precisar ser ajustado para o seu ambiente
DATA_FOLDER = Path(r"C:\Users\joao.beneton\Downloads\New project\data - Copia") 

# --- CARREGAMENTO ÚNICO DOS COMPONENTES ---
@st.cache_resource
def load_shared_components():
    print(">> App Principal: Carregando componentes...")
    llm = ChatOpenAI(model_name=get_config('model_name'), temperature=0, openai_api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
    
    vectorstore = FAISS.load_local("faiss_index_limpo", embeddings, allow_dangerous_deserialization=True)
    with open("bm25_index_limpo.pkl", "rb") as f:
        bm25_retriever_full = pickle.load(f)
    with open("ordered_chunks.pkl", "rb") as f:
        ordered_chunks = pickle.load(f)

    print(">> App Principal: Componentes carregados e cacheados.")
    return llm, vectorstore, bm25_retriever_full, ordered_chunks

@st.cache_data
def get_available_norms(data_folder: Path) -> List[str]:
    print(">> App Principal: Buscando lista de normas disponíveis")
    if not data_folder.exists():
        st.error(f"A pasta de dados especificada não foi encontrada: {data_folder}")
        return []
    norm_files = [f.name for f in data_folder.rglob("*") if f.suffix.lower() in [".pdf", ".md", ".markdown"]]
    return sorted(norm_files)

def save_feedback(file_path: str, question: str, answer: str, motor: str):
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    data.append({"question": question, "answer": answer, "motor_utilizado": motor})
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    st.set_page_config(page_title="Chatbots BACEN", layout="wide")
    st.title("Chatbots para Consulta de Normas do BACEN")
    st.caption(f"Utilizando o modelo `{get_config('model_name')}` para análise.")

    llm, vectorstore, bm25_retriever, ordered_chunks = load_shared_components()
    available_norms = get_available_norms(DATA_FOLDER)

    with st.sidebar:
        st.header("⚙️ Selecione o Motor do Chatbot")
        
        # --- LISTA DE MOTORES ATUALIZADA ---
        motor_selecionado = st.radio(
            "Escolha a versão do pipeline de RAG:",
            (
                "Motor Unificado (Recomendado)", # <-- NOVA OPÇÃO
                "Conselho de Especialistas (v2+v3+v4 + Juiz)", 
                "Foco Específico (Seleção Manual)", 
                "Híbrido v1.0 (Ensemble + Widening)", 
                "Híbrido v2.0 (HyDE + Foco)",
                "Otimizado 3.0 (Rewrite + Refine)"
            ),
            index=0, 
            key="motor_chatbot"
        )
        st.info(f"Você selecionou o motor: **{motor_selecionado}**")

        normas_selecionadas = []
        if motor_selecionado == "Foco Específico (Seleção Manual)":
            st.markdown("---")
            st.subheader("Filtrar Normas")
            if not available_norms:
                 st.error("Nenhuma norma encontrada na pasta de dados para seleção.")
            else:
                st.warning("Selecione uma ou mais normas para focar a busca.")
                normas_selecionadas = st.multiselect(
                    "Normas disponíveis:",
                    options=available_norms,
                    key="norm_multiselect"
                )

        if st.button("Limpar Histórico do Chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Loop que exibe o histórico de mensagens
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # --- LÓGICA DE AVALIAÇÃO ATUALIZADA ---
            # Verifica se a mensagem é do assistente e se o motor é um dos principais
            if (message["role"] == "assistant" and 
                message.get("motor") in ["Motor Unificado (Recomendado)", "Conselho de Especialistas (v2+v3+v4 + Juiz)"] and
                idx > 0): # Garante que há uma pergunta de usuário antes para associar

                with st.expander("Avaliar esta resposta"):
                    col1, col2 = st.columns(2)
                    
                    pergunta_correspondente = st.session_state.messages[idx - 1]['content']

                    with col1:
                        if st.button("👍 Correta", key=f"correta_{idx}", use_container_width=True):
                            save_feedback(
                                file_path="respostas_corretas.json",
                                question=pergunta_correspondente,
                                answer=message["content"],
                                motor=message.get("motor")
                            )
                            st.toast("Obrigado! Resposta salva como correta.", icon="✅")
                    with col2:
                        if st.button("👎 Incorreta", key=f"incorreta_{idx}", use_container_width=True):
                            save_feedback(
                                file_path="respostas_incorretas.json",
                                question=pergunta_correspondente,
                                answer=message["content"],
                                motor=message.get("motor")
                            )
                            st.toast("Obrigado! Resposta marcada para análise.", icon=" flagged")

    if prompt := st.chat_input("Digite sua pergunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if motor_selecionado == "Foco Específico (Seleção Manual)" and not normas_selecionadas:
            st.warning("Por favor, selecione ao menos uma norma na barra lateral para usar o motor de Foco Específico.")
            st.stop()

        with st.spinner(f"Analisando normas com o motor '{motor_selecionado}'..."):
            result = {}
            
            history_for_memory = st.session_state.messages[:-1]
            
            # --- CADEIA DE IF/ELIF ATUALIZADA ---
            if motor_selecionado == "Motor Unificado (Recomendado)":
                result = obter_resposta_unificada(
                    question=prompt, 
                    chat_history=history_for_memory,
                    llm=llm, 
                    vectorstore=vectorstore, 
                    bm25_retriever_full=bm25_retriever, 
                    ordered_chunks=ordered_chunks,
                    available_norms=available_norms
                )
            elif motor_selecionado == "Conselho de Especialistas (v2+v3+v4 + Juiz)":
                result = obter_resposta_conselho(
                    question=prompt, 
                    chat_history=history_for_memory,
                    llm=llm, 
                    vectorstore=vectorstore, 
                    bm25_retriever_full=bm25_retriever, 
                    ordered_chunks=ordered_chunks,
                    available_norms=available_norms
                )
            elif motor_selecionado == "Foco Específico (Seleção Manual)":
                result = obter_resposta_v5(
                    question=prompt,
                    llm=llm, 
                    vectorstore=vectorstore, 
                    bm25_retriever_full=bm25_retriever, 
                    ordered_chunks=ordered_chunks,
                    normas_selecionadas=normas_selecionadas
                )
            elif motor_selecionado == "Híbrido v1.0 (Ensemble + Widening)":
                result = obter_resposta_v2(question=prompt, llm=llm, vectorstore=vectorstore, bm25_retriever_full=bm25_retriever, ordered_chunks=ordered_chunks)
            elif motor_selecionado == "Híbrido v2.0 (HyDE + Foco)":
                result = obter_resposta_v4(question=prompt, llm=llm, vectorstore=vectorstore, bm25_retriever_full=bm25_retriever, ordered_chunks=ordered_chunks)
            elif motor_selecionado == "Otimizado 3.0 (Rewrite + Refine)":
                result = obter_resposta_v3(question=prompt, llm=llm, vectorstore=vectorstore, bm25_retriever_full=bm25_retriever)
            
            resposta = result.get("answer", "Ocorreu um erro ao gerar a resposta.")
            st.session_state.messages.append({"role": "assistant", "content": resposta, "motor": motor_selecionado})
            st.rerun()

if __name__ == "__main__":
    main()