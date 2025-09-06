from typing import List, Sequence

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, MessageGraph

from chains import generate_chain, reflect_chain


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue,  {END:END, REFLECT:REFLECT} )
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
#print(graph.get_graph().draw_mermaid())
#graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = HumanMessage(content="""
        Torne o texto abaixo mais humanizado:
        A Inteligência Artificial (IA) é um dos campos mais dinâmicos e transformadores da ciência da computação.
        Ela pode ser entendida como a capacidade de sistemas computacionais realizarem tarefas que normalmente exigiriam
        inteligência humana, como raciocínio lógico, reconhecimento de padrões, aprendizado, tomada de decisão e até mesmo
        criatividade.
                                  """)
    response = graph.invoke(inputs)

    # Imprime a última resposta gerada pela IA
    last_ai = next((m for m in reversed(response) if isinstance(m, AIMessage)), None)
    if last_ai:
        print("Última resposta da IA:", last_ai.content)
    last_ai_message = next((m for m in reversed(response) if getattr(m, "type", None) == "ai"), None)
    