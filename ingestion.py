import os
import warnings
from dotenv import load_dotenv
from langchain_core._api.deprecation import (
    LangChainDeprecationWarning,
    LangChainPendingDeprecationWarning,
)

# Silencia avisos de depreciação do LangChain (Pydantic v1 shim etc.)
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

# Importações que podem emitir warnings no import devem vir após o filtro.
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import TextLoader

load_dotenv()

if __name__ == '__main__':
    print("Ingestão simples: carregar texto -> gerar embeddings -> enviar ao Pinecone")

    # 1) Carregar variáveis de ambiente necessárias
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    hf_model = os.getenv("HF_EMBED_MODEL", "BAAI/bge-large-en-v1.5")  # 1024-d
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index = os.getenv("PINECONE_INDEX")

    if not hf_token:
        print("Defina HUGGINGFACEHUB_API_TOKEN no .env")
        raise SystemExit(1)
    if not pinecone_api_key or not pinecone_index:
        print("Defina PINECONE_API_KEY e PINECONE_INDEX no .env")
        raise SystemExit(1)

    # 2) Carregar o arquivo de texto
    print("Carregando documento...")
    loader = TextLoader("mediumblog1.txt", encoding="utf-8")
    docs = loader.load()
    print(f"Documentos carregados: {len(docs)}")

    # 3) Quebrar em pedaços (chunks) simples
    print("Fatiando em chunks...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Chunks criados: {len(chunks)}")

    # 4) Criar o objeto de embeddings (HF Inference API)
    print(f"Gerando embeddings com o modelo: {hf_model}")
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=hf_token,
        model=hf_model,
    )

    # Dica didática: garanta que o índice Pinecone tem a MESMA dimensão do modelo de embeddings.
    # Ex.: BAAI/bge-large-en-v1.5 -> 1024 dimensões.

    # 5) Enviar para o Pinecone (índice deve existir previamente com a dimensão correta)
    print(f"Enviando {len(chunks)} chunks para o índice '{pinecone_index}' no Pinecone...")
    try:
        # Compatível com langchain-pinecone 0.2.x e 0.3.x
        texts = [d.page_content for d in chunks]
        metadatas = [getattr(d, "metadata", {}) for d in chunks]
        PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            index_name=pinecone_index,
        )
        print("Pronto! Upsert concluído no Pinecone.")
    except Exception as e:
        # Mensagem simples e direta
        print("Falha ao enviar ao Pinecone:", str(e).splitlines()[0])
