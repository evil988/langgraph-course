"""
Arquivo: react.py

Objetivo
- Definir ferramentas (tools) que o agente poderá usar
- Configurar o LLM (modelo de linguagem) e vinculá-lo às ferramentas

Este arquivo fica focado na configuração do ambiente ReAct:
- ReAct (Reason + Act) é um padrão em que o modelo "pensa" (razão) e
  "age" (executa ferramentas) de forma iterativa para resolver tarefas.
"""

# Carrega variáveis de ambiente do arquivo .env (ex.: chaves de API)
from dotenv import load_dotenv

# Decorador e tipos utilitários para declarar ferramentas no LangChain
from langchain_core.tools import tool

# Ferramenta de busca web (Tavily) integrada ao LangChain
from langchain_tavily import TavilySearch

# Cliente do LLM hospedado pela Groq (usa modelos como Llama 3.1)
from langchain_groq import ChatGroq

# Lê o arquivo .env (se existir) e injeta as variáveis no processo
load_dotenv()

@tool
def triple(num: float) -> float:
    """
    Tool: triple

    Função/ferramenta simples que triplica um número. Ao usar o
    decorador @tool, ela passa a poder ser chamada pelo LLM durante
    a execução do grafo (quando o modelo "decide agir").

    Parâmetros
    - num: número (int/float) a ser triplicado.

    Retorno
    - float: o valor de entrada multiplicado por 3.
    """
    return float(num) * 3

# Lista de ferramentas disponíveis para o agente.
# - TavilySearch: realiza busca web; aqui limitada a 1 resultado para ser objetiva.
# - triple: nossa função definida acima para triplicar números.
tools = [TavilySearch(max_results=1), triple]

# Instancia o LLM da Groq.
# Requisitos:
# - Definir a variável de ambiente `GROQ_API_KEY` (por isso o load_dotenv acima).
# Parâmetros principais:
# - model: nome do modelo (ex.: "llama-3.3-70b-versatile").
# - temperature: 0 para respostas determinísticas (útil em agentes/ferramentas).
# Em seguida, `bind_tools(tools)` habilita o LLM a chamar as ferramentas declaradas.
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).bind_tools(tools)
