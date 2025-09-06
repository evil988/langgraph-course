from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um revisor especializado em humanizar textos. "
            "Produza apenas feedback acionável para tornar o texto mais natural, fluido e próximo da escrita humana. "
            "Regras: 1) Avalie somente o texto fornecido; 2) Não reescreva o texto completo; "
            "3) Foque em clareza, tom, concisão e coesão; 4) Dê exemplos curtos quando útil; "
            "5) Sempre forneça pelo menos 3 sugestões, mesmo que pequenas.",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "human",
            "Analise apenas a ÚLTIMA mensagem do assistente na conversa acima. Responda no formato:\n"
            "- Problemas observados: <bullets curtos>\n"
            "- Sugestões de melhoria: <bullets com ações e exemplos breves>\n"
            "- Observações de tom/estilo (1–2 linhas)"
        ),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um escritor especializado em produzir textos sobre diversos assuntos de forma objetiva e concisa, sempre atendendo às solicitações do usuário. "
            "Caso o usuário apresente críticas ou comentários, responda reformulando a versão anterior de maneira aprimorada, incorporando o feedback recebido. ",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


#llm = ChatOpenAI()
llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.0)
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm
