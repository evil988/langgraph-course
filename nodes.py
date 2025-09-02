"""
Nós definimos aqui os "nós" (nodes) usados no grafo do LangGraph.

Este arquivo expõe:
- `run_agent_reasoning`: um nó de raciocínio que chama o LLM com uma
  mensagem de sistema + o histórico de mensagens do estado.
- `tool_node`: um nó pré-construído do LangGraph responsável por executar
  ferramentas (tools) quando o LLM solicitar `tool_calls` na resposta.
"""

from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langsmith import traceable

# Importamos o LLM e a lista de ferramentas definidos em `react.py`.
# - `llm`: instância do modelo (por exemplo, OpenAI, Anthropic, etc.)
# - `tools`: lista/registry de funções/tool wrappers que o agente pode usar
from react import llm, tools

# Carrega variáveis de ambiente do arquivo `.env` (chaves de API, etc.).
load_dotenv()

# Mensagem de sistema enviada ao modelo em toda a execução para orientar
# o comportamento do assistente (instruções de alto nível).
SYSYEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""


@traceable(name="agent_reasoning")
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """Executa o nó de raciocínio do agente.

    Parâmetros
    - state (MessagesState): estado do LangGraph contendo o histórico de
      mensagens sob a chave "messages" (ex.: usuário, assistente, tool, etc.).

    Retorno
    - MessagesState: um dicionário com a chave "messages" contendo a nova
      mensagem gerada pelo LLM (normalmente um `AIMessage`).
    """

    # Preparamos o "prompt" do modelo como uma lista de mensagens ao estilo
    # chat: começamos com a mensagem de sistema e, em seguida, adicionamos
    # todo o histórico contido no estado.
    messages = [{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]]

    # Chamamos o LLM. A implementação de `llm.invoke` (definida em `react.py`)
    # deve aceitar uma lista de mensagens e retornar um objeto de mensagem
    # do assistente (por exemplo, `AIMessage`).
    response = llm.invoke(messages)

    # O LangGraph espera que cada nó retorne um dicionário parcial do estado.
    # Aqui, retornamos uma lista com a nova mensagem para ser anexada ao
    # histórico global.
    return {"messages": [response]}


# Nó pré-construído do LangGraph que executa ferramentas quando o LLM
# produz `tool_calls`. Ele encontra a ferramenta pelo nome, chama com os
# argumentos e devolve a(s) mensagem(ns) de resultado para o grafo.
tool_node = ToolNode(tools)
