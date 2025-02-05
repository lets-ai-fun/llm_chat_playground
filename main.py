from langchain_community.document_loaders.web_base import WebBaseLoader
import bs4

from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.messages import AIMessage
import vectorestore_api
from my_loader import MyLoader
import chain_api
import doc_loader_api
from my_types import CHAIN_TYPE

from langchain.globals import set_verbose, set_debug
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables import RunnableSequence



def setup_vectorstore(embeddings: Embeddings) -> VectorStore:
    # the loader used are
    # - 1 WebBaseLoader to scrape content from url
    # - n docuemnt loader, one for each docuemnt in 'documents' folder
    urls = ['https://lilianweng.github.io/posts/2023-06-23-agent/']
    webLoader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
    )
    doc_loader = doc_loader_api.get_loaders(doc_folder="./documents", suffix="pdf")
    # note:
    # - if you don't want to use WebBaseLoader and only populate vector store with docuemnts in 'documents' folder
    #   update the code to 'loaders = doc_loader'
    # - if you want to use the WebBaseLoader only to populate vector store using only the content scaped from provided url
    #   update the code to 'loaders = [MyLoader(loader=webLoader, name=urls[0])]'
    loaders = doc_loader + [MyLoader(loader=webLoader, name=urls[0])]
    vectorestore = vectorestore_api.setup_vectorstore(
        loader=loaders,
        db_folder="./db",
        embeddings=embeddings,
        chunk_size=1000,
        chunk_overlap=500
    )
    print(f"vectorstore setup: embeddings size {len(vectorestore_api.__get_stored_documents(vectorestore))}")
    return vectorestore


def load_vectorstore(embeddings: Embeddings) -> VectorStore:
    vectorstore = vectorestore_api.load_vectorstore(
        db_folder="./db",
        embeddings=embeddings
    )
    print(f"vectorstore load: embeddings size {len(vectorestore_api.__get_stored_documents(vectorstore))}")
    return vectorstore

#enable llm debug log -> set to False or comment line if you don't want this log(s)
set_debug(True)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

input_erase_vectorstore = input("Do you want to reset docstore? [y/n]")
vectorstore: VectorStore = None
if 'y' == input_erase_vectorstore:
    vectorstore = setup_vectorstore(embeddings= embeddings)
else:
    vectorstore = load_vectorstore(embeddings= embeddings)

input_chain_to_invoke= input(
    """Which chain do you want to use [ 'quit' to exit]?
- 'simple' for simple chain with llm invocation using user input
- 'rag' for chain using RAG on documents provided
"""
)
chain_type: CHAIN_TYPE
chain: RunnableSequence
match input_chain_to_invoke:
    case 'simple':
        chain_type = CHAIN_TYPE.CHAIN_SIMPLE
        chain = chain_api.setup_chain(
            ollama_model_name="llama3.2"
        )
    case 'rag':
        chain_type = CHAIN_TYPE.CHAIN_RAG
        chain = chain_api.setup_rag_chain(
            retriever=vectorstore.as_retriever(),
            ollama_model_name="llama3.2"
        )

chain_api.endless_chat(chain= chain, chain_type= chain_type)

