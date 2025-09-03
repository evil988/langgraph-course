"""
Arquivo: react.py

Objetivos
- Declarar ferramentas (tools) disponíveis ao agente.
- Configurar o LLM e vinculá‑lo às ferramentas no estilo ReAct.

Contexto
- ReAct (Reason + Act) alterna raciocínio e ação: o modelo "pensa"
  e, quando necessário, "age" chamando ferramentas de forma iterativa
  para resolver uma tarefa.
"""

# Carrega variáveis de ambiente do arquivo .env (ex.: chaves de API).
from dotenv import load_dotenv
import os
from typing import Optional, List, Dict, Any
import warnings

# Importa classes de warning e aplica filtro ANTES das integrações que emitem avisos no import
from langchain_core._api.deprecation import (
    LangChainDeprecationWarning,
    LangChainPendingDeprecationWarning,
)
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

# Decorador e tipos utilitários para declarar ferramentas no LangChain.
from langchain_core.tools import tool

# Ferramenta de busca web (Tavily) integrada ao LangChain.
# Observação: requer `TAVILY_API_KEY` definido no ambiente.
from langchain_tavily import TavilySearch

# Cliente do LLM hospedado pela Groq (modelos Llama 3.x).
# Aqui utilizamos `llama-3.3-70b-versatile`.
from langchain_groq import ChatGroq

# Vetor store (Pinecone) e embeddings (HuggingFace)
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Lê o arquivo .env (se existir) e injeta as variáveis no processo.
load_dotenv()

@tool
def triple(num: float) -> float:
    """
    Triplica um número

    Parâmetros
    - num: número (int/float) a triplicar.

    Retorno
    - float: valor de entrada multiplicado por 3.
    """
    return float(num) * 3


def _get_vectorstore(namespace: Optional[str] = None) -> PineconeVectorStore:
    """
    Cria um PineconeVectorStore com embeddings da HuggingFace Inference API.

    Requisitos (.env):
    - HUGGINGFACEHUB_API_TOKEN: token de acesso à HuggingFace
    - HF_EMBED_MODEL (opcional): nome do modelo de embedding (padrão: BAAI/bge-large-en-v1.5)
    - PINECONE_INDEX: nome do índice no Pinecone que já deve existir com a dimensão correta
    - PINECONE_API_KEY: chave do Pinecone (lida pelo client do Pinecone/LC)
    """
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    hf_model = os.getenv("HF_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    index_name = os.getenv("PINECONE_INDEX")

    if not hf_token:
        raise ValueError("Defina HUGGINGFACEHUB_API_TOKEN no .env para usar a busca semântica.")
    if not index_name:
        raise ValueError("Defina PINECONE_INDEX no .env para usar a busca semântica.")

    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=hf_token,
        model=hf_model,
    )

    # Observação: o namespace é opcional e útil para segmentar dados no mesmo índice.
    return PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)


@tool("similarity_search")
def similarity_search(
    query: str,
    k: int = 5,
    namespace: Optional[str] = None,
    filter: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    """
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer conceptual questions. "
    "Input should be a search query."

    Parâmetros
    - query: texto da consulta.
    - k: quantidade de resultados (top-k).
    - namespace: (opcional) restringe a um namespace específico no índice.
    - filter: (opcional) dicionário de filtros por metadados.

    Retorno
    - Lista de dicts com campos: {"content", "metadata", "score"},
      formato adequado para o LLM consumir e citar fontes.
    """
    vs = _get_vectorstore(namespace=namespace)

    # Alguns releases aceitam filter no método; garantimos compatibilidade.
    try:
        results = vs.similarity_search_with_score(query, k=k, filter=filter)  # type: ignore[arg-type]
    except TypeError:
        results = vs.similarity_search_with_score(query, k=k)

    output: List[Dict[str, Any]] = []
    for doc, score in results:
        output.append(
            {
                "content": doc.page_content,
                "metadata": getattr(doc, "metadata", {}) or {},
                # O valor exato depende da métrica do índice; retornamos bruto.
                "score": float(score) if isinstance(score, (int, float)) else score,
            }
        )
    return output

# Lista inicial de ferramentas disponíveis ao agente.
# - TavilySearch: busca web (limitada a 1 resultado para objetividade).
# - triple: função simples para triplicar números.
tools = [TavilySearch(max_results=1), similarity_search, triple]

# Instancia o LLM da Groq e vincula as ferramentas.
# Requisitos: variável de ambiente `GROQ_API_KEY` definida (via .env).
# Principais parâmetros:
# - model: nome do modelo (ex.: "llama-3.3-70b-versatile").
# - temperature: 0 para respostas determinísticas (útil com ferramentas).
# `bind_tools(tools)` habilita o LLM a chamar as ferramentas declaradas.
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).bind_tools(tools)
