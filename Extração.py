import os
from pathlib import Path
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared, operations
from unstructured_client.models.errors import SDKError
from dotenv import load_dotenv

# --- Configuração ---
load_dotenv()
UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")
if not UNSTRUCTURED_API_KEY:
    raise ValueError("Erro. Verifique seu arquivo .env.")

BASE_PATH = Path(r"C:\Users\joao.beneton\Downloads\New project - modificação livro")
PDF_PATH = BASE_PATH / "Normas" / "Res_2836_v3_L.pdf"

OUTPUT_DIR = BASE_PATH / "Normas - Markdown"
OUTPUT_MD_PATH = OUTPUT_DIR / "Res_2836_RAW.md"

def extrair_markdown_do_pdf(pdf_arquivo, md_saida):

    client = UnstructuredClient(api_key_auth=UNSTRUCTURED_API_KEY)
    print(f"Iniciando a extração do arquivo: {pdf_arquivo}")

    try:
        with open(pdf_arquivo, "rb") as f:
            files = shared.Files(
                content=f.read(),
                file_name=pdf_arquivo.name,
                )
            
            # requisição para a API
            req = operations.PartitionRequest(
                partition_parameters=shared.PartitionParameters(
                files=files,
                strategy="hi_res",
                hi_res_model_name="yolox",
                pdf_infer_table_structure=True,
                languages=["por"],
                output_format="application/json",))
    
            # Envia a requisição e obtém a resposta
            print("Enviando para a API da Unstructured.io")
            resp = client.general.partition(request=req)

        # Processa a resposta
        if resp.elements:
            print("Extração concluída.")
            with open(md_saida, "w", encoding="utf-8") as f:
                for element in resp.elements:
                    f.write(str(element.get("text", "")))
                    f.write("\n\n")
            print(f"Arquivo Markdown bruto salvo em: {md_saida}")
        else:
            print("Erro: A API não retornou elementos.")

    except SDKError as e:
        print(f"Erro na API da Unstructured: {e}")
    except FileNotFoundError:
        print(f"Erro: Arquivo PDF não encontrado em: {pdf_arquivo}")
    except Exception as e:
        print(f"Um erro inesperado ocorreu: {e}")

if __name__ == "__main__":
    if not PDF_PATH.exists():
        print(f"Aviso: O arquivo PDF de entrada não foi encontrado no caminho:")
        print(f"{PDF_PATH}")
        print("Por favor, verifique o caminho e tente novamente.")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Diretório de saída verificado/criado em: {OUTPUT_DIR}")
        
        extrair_markdown_do_pdf(PDF_PATH, OUTPUT_MD_PATH)