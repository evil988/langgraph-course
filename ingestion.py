import os

# Set a user agent before importing clients that might warn on missing value
os.environ.setdefault("USER_AGENT", "langgraph-course/ingestion")

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

#text="""Agent memory refers to the capability of artificial intelligence agents to store, recall, and use information from past interactions in order to guide future behavior. Instead of treating each interaction as isolated, memory-enabled agents can accumulate knowledge, learn preferences, and adapt over time. This concept is inspired by human cognition, where memory serves as a foundation for reasoning, decision-making, and personal continuity.
#There are generally three layers of agent memory. Short-term memory captures immediate context within a single interaction, such as the last user query or a temporary goal. Long-term memory stores knowledge across multiple sessions, enabling the agent to remember facts, user preferences, or previous decisions. Episodic memory focuses on specific past events, allowing the agent to recall detailed scenarios and experiences. Together, these layers make interactions more coherent and personalized.
#Technically, agent memory can be implemented through structured databases, vector embeddings for semantic search, or hybrid systems that combine symbolic reasoning with machine learning. Vector-based memory, in particular, allows agents to retrieve relevant information using similarity measures rather than exact keyword matches, making recall more flexible and closer to human-like understanding.
#The integration of memory into intelligent agents opens possibilities in diverse fields: virtual assistants that remember user routines, customer support bots that build long-term relationships with clients, or autonomous systems that refine strategies based on accumulated experience. However, challenges remain, especially concerning privacy, scalability, and the risk of biased or outdated information persisting in memory.
#In short, agent memory transforms artificial agents from reactive tools into adaptive systems capable of continuity, personalization, and context-aware reasoning. It is a critical step toward building AI that feels not just intelligent, but genuinely attentive."""
#docs_list = [Document(page_content=text, metadata={"source": "manual"})]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

model_id = "BAAI/bge-small-en-v1.5"

# Use local sentence-transformers embeddings to avoid endpoint timeouts
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_id,
    # Keep on CPU by default; adjust to "cuda" if desired
    model_kwargs={"device": "cpu"},
    # Normalize for cosine similarity retrieval consistency
    encode_kwargs={"normalize_embeddings": True},
)

Chroma(collection_name="rag-chroma",persist_directory="./.chroma", embedding_function=hf_embeddings).reset_collection()

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    # embedding=OpenAIEmbeddings(),
    embedding=hf_embeddings,
    persist_directory="./.chroma",
)

# Export a retriever used by the graph nodes
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=hf_embeddings,
).as_retriever()

if __name__ == "__main__":
    print("Built Chroma vectorstore with", len(doc_splits), "chunks → ./.chroma")
