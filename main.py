"""
Arquivo: main.py

Objetivo
- Construir o grafo ReAct (Reason + Act) usando LangGraph,
  conectando um nó de raciocínio (LLM) e um nó de execução de ferramentas.
- Definir a lógica de continuação que decide quando parar ou executar uma tool.
- Compilar e (opcionalmente) renderizar o grafo em "flow.png" para visualização.
"""

import os
from dotenv import load_dotenv

# Tipos/utilitários de mensagens e estado do LangChain/LangGraph
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

# Nossos nós definidos em `nodes.py`: o nó de raciocínio e o nó de ferramentas
from nodes import run_agent_reasoning, tool_node

# Carrega variáveis de ambiente do arquivo .env (ex.: chaves de API)
load_dotenv()

# Habilita LangSmith/LangChain tracing automaticamente quando uma API key existir.
# Mantém compatibilidade com variáveis antigas do LangChain.
if os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "langgraph-course"))

# Nomes (labels) para os nós do grafo. Esses labels são usados ao registrar
# nós e ao criar arestas entre eles.
AGENT_REASON = "agent_reason"  # nó de raciocínio (chama o LLM)
ACT = "act"                    # nó que executa ferramentas (ToolNode)

# Índice -1 facilita pegar a última mensagem da lista de mensagens no estado.
LAST = -1


def should_continue(state: MessagesState) -> str:
    """Decide o próximo passo a partir da última resposta do LLM.

    Se a última mensagem NÃO contiver `tool_calls`, encerramos o fluxo (END).
    Caso contrário, seguimos para o nó de execução de ferramentas (ACT).
    """
    # A propriedade `tool_calls` é preenchida quando o LLM solicita a execução
    # de alguma tool registrada. Se estiver vazia/ausente, podemos encerrar.
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


# Criamos um grafo de estados baseado em mensagens (MessagesState). Isso permite
# que cada nó acrescente novas mensagens ao histórico de conversa.
flow = StateGraph(MessagesState)

# Registra o nó de raciocínio (LLM) e define-o como ponto de entrada do grafo.
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)

# Registra o nó de execução de ferramentas (pré-construído em `nodes.py`).
flow.add_node(ACT, tool_node)

# Arestas condicionais a partir do nó de raciocínio:
# - Se `should_continue` retornar END, parar.
# - Se retornar ACT, ir para o nó de ferramentas.
flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {END: END, ACT: ACT},
)

# Após executar uma ferramenta, voltamos ao nó de raciocínio para o próximo passo
# (loop ReAct). Isso permite múltiplas iterações pensar->agir->pensar.
flow.add_edge(ACT, AGENT_REASON)

# Compila o grafo em um aplicativo executável. Também renderizamos uma imagem do
# grafo em formato Mermaid/PNG para facilitar o entendimento do fluxo.
app = flow.compile()

# Configure metadados de execução para facilitar o filtro no LangSmith
app = app.with_config({
    "run_name": "react_graph",
    "tags": ["react", "langgraph-course"],
})
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


if __name__ == "__main__":
    # Exemplo rápido de uso: fazemos uma pergunta que induz o agente a
    # pesquisar a temperatura em Tóquio e depois triplicar o valor.
    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke(
        {
            "messages": [
                HumanMessage(
                    #content="What is the temperature in Tokyo? List it and then triple it"
                    content="What is Pinecone and machine Learning?"
                    #content="Quando é 3 vezes 10?"
                    #content="Qual a temperatura em Tokyo? O que é Pinecone? O que é Machine Learning? Quando é 3 vezes 10?"
                )
            ]
        }
    )

    # Exibimos o conteúdo da última mensagem produzida pelo fluxo.
    print(res["messages"][LAST].content)
