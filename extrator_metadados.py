import json
from typing import Dict, Any

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from configs_v2 import METADATA_EXTRACTOR_TEMPLATE

class ExtratorDeMetadados:

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt_template = PromptTemplate.from_template(METADATA_EXTRACTOR_TEMPLATE)
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def extrair_filtros(self, question: str) -> Dict[str, Any]:

        print(f"-> Extraindo filtros da pergunta: '{question}'")
        try:
            response_str = self.chain.invoke({"question": question})
            if "```json" in response_str:
                response_str = response_str.split("```json\n")[1].split("```")[0]
            
            filter_dict = json.loads(response_str)
            
            cleaned_filter_dict = {k: v for k, v in filter_dict.items() if v}

            if cleaned_filter_dict:
                print(f"--> Filtros extraídos: {cleaned_filter_dict}")
            else:
                print("--> Nenhum filtro relevante extraído.")

            return cleaned_filter_dict
        except (json.JSONDecodeError, IndexError) as e:
            print(f"⚠️ Aviso: Não foi possível decodificar a resposta JSON do extrator. Erro: {e}")
            print(f"Resposta recebida: {response_str}")
            return {} 
        except Exception as e:
            print(f"Erro inesperado no ExtratorDeMetadados: {e}")
            return {}