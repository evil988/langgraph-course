from langchain import hub
from langchain_core.output_parsers import StrOutputParser
#from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

#llm = ChatOpenAI(temperature=0)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
