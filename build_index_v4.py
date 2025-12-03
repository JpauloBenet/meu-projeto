import os
import pickle
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

DATA_FOLDER = Path(r"C:\Users\joao.beneton\Downloads\New project\data - Copia")

def extract_metadata_from_filename(file: Path):
    """Extrai metadados do nome do arquivo de forma robusta."""
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
        "contexto":r"^(?:contexto_?)(\d+)",
    }
    for norma_type, pattern in patterns.items():
        m = re.match(pattern, name_lower)
        if m:
            meta["tipo_norma"] = norma_type
            meta["numero_norma"] = m.group(1)
            return meta
    print(f"⚠️  Aviso: O arquivo '{file.name}' não corresponde a nenhum padrão de metadados e será ignorado.")
    return {}

def build_and_save_indexes(faiss_path: str, bm25_path: str, ordered_chunks_path: str):
    """Constrói e salva todos os índices necessários para o chatbot."""
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""])
    
    headers_to_split_on = [("#", "Titulo"), ("##", "Artigo"), ("###", "Paragrafo")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

    all_docs = []
    for file_path in DATA_FOLDER.rglob("*"):
        suf = file_path.suffix.lower()
        if not (suf in [".pdf", ".md", ".markdown"]): continue

        print(f"📄 Processando arquivo: {file_path.name}")
        file_meta = extract_metadata_from_filename(file_path)
        if not file_meta: continue

        file_meta["formato"] = suf.strip(".")
        file_meta["origem"] = file_path.name
        
        if suf == ".pdf":
            docs = PyMuPDFLoader(str(file_path)).load()
            chunks = recursive_splitter.split_documents(docs)
        elif suf in (".md", ".markdown"):
            raw_text = file_path.read_text(encoding="utf-8")
            chunks = markdown_splitter.split_text(raw_text)
        
        for chunk in chunks: chunk.metadata.update(file_meta)
        all_docs.extend(chunks)
    
    chunks_filtrados = [doc for doc in all_docs if len(doc.page_content.strip()) > 50]
    print(f"🔍 Chunks totais após filtragem: {len(chunks_filtrados)}")
    if not chunks_filtrados:
        raise RuntimeError("Nenhum documento encontrado e processado. Verifique a pasta e os nomes dos arquivos.")

    for i, doc in enumerate(chunks_filtrados):
        doc.metadata['original_index'] = i

    texts_com_metadata = []
    metadatas = []
    for d in chunks_filtrados:
        artigo = d.metadata.get("Artigo", "").replace("#", "").strip()
        paragrafo = d.metadata.get("Paragrafo", "").replace("#", "").strip()
        
        context_header = f"[Norma: {d.metadata.get('tipo_norma', 'N/A').title()} {d.metadata.get('numero_norma', 'N/A')}"
        if artigo:
            context_header += f" | {artigo}"
        if paragrafo:
            context_header += f" | {paragrafo}"
        context_header += "]\n"

        texts_com_metadata.append(context_header + d.page_content)
        metadatas.append(d.metadata)

    embedder = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=API_KEY, chunk_size=256)
    vectorstore = FAISS.from_texts(texts=texts_com_metadata, embedding=embedder, metadatas=metadatas, normalize_L2=True)
    vectorstore.save_local(faiss_path)
    print(f"✅ FAISS reconstruído e salvo em `{faiss_path}`.")

    bm25_retriever = BM25Retriever.from_documents(chunks_filtrados)
    with open(bm25_path, "wb") as f: pickle.dump(bm25_retriever, f)
    print(f"✅ BM25 reconstruído e salvo em `{bm25_path}`.")

    with open(ordered_chunks_path, "wb") as f: pickle.dump(chunks_filtrados, f)
    print(f"✅ Lista ordenada de chunks salva em `{ordered_chunks_path}`.")

if __name__ == "__main__":
    build_and_save_indexes(
        faiss_path="faiss_index_limpo",
        bm25_path="bm25_index_limpo.pkl",
        ordered_chunks_path="ordered_chunks.pkl"
    )