from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables import RunnableSequence
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders.web_base import WebBaseLoader

import bs4

import my_types
import vectorestore_api
import doc_loader_api
import chain_api
from my_loader import MyLoader


def __setup_vectorstore(embeddings: Embeddings) -> VectorStore:
    """
    Setup vector store content using document loaders.
    Previous content is erased before loading new one
    """
    # the loader used are
    # - 1 WebBaseLoader to scrape content from url(s) specifien in RAG_URLS constant
    # - n docuemnt loader, one for each docuemnt in 'documents' folder
    urls = my_types.RAG_URLS
    webLoader: WebBaseLoader= None
    if len(urls) > 0:
        print(f"found url(s) to load into vector store {urls}")
        webLoader = WebBaseLoader(
            web_paths=urls,
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
        )

    # document loaders fro docuemnts in 'documents' folder
    doc_loaders = doc_loader_api.get_loaders(doc_folder="./documents", suffix="pdf")

    if None != webLoader:
        loaders = doc_loaders + [MyLoader(loader=webLoader, name=urls[0])]
    else:
        loaders = doc_loaders

    vectorestore = vectorestore_api.setup_vectorstore(
        loader=loaders,
        db_folder="./db",
        embeddings=embeddings,
        chunk_size=1000,
        chunk_overlap=500
    )
    print(f"vectorstore setup: embeddings size {len(vectorestore_api.__get_stored_documents(vectorestore))}")
    return vectorestore


def __load_vectorstore(embeddings: Embeddings) -> VectorStore:
    """
    Setup vector store by loading its content form a folder
    were it was previously persisted.
    """
    vectorstore = vectorestore_api.load_vectorstore(
        db_folder="./db",
        embeddings=embeddings
    )
    print(f"vectorstore load: embeddings size {len(vectorestore_api.__get_stored_documents(vectorstore))}")
    return vectorstore


def start_program_ui() -> (RunnableSequence, my_types.CHAIN_TYPE):
    """
    Start program flow text UI:
    - asks user about using a persisted vector store o create a new one
    - asks user which tchain to use: simple or rag
    """
    embeddings = OllamaEmbeddings(model=my_types.OLLAMA_EMBEDDINGS_NAME)
    input_erase_vectorstore = input("Do you want to reset docstore? [y/n]")
    vectorstore: VectorStore = None
    if 'y' == input_erase_vectorstore:
        vectorstore = __setup_vectorstore(embeddings=embeddings)
    else:
        vectorstore = __load_vectorstore(embeddings=embeddings)
    input_chain_to_invoke = input("""Which chain do you want to use [ 'quit' to exit]?
- 'simple' for simple chain with llm invocation using user input
- 'rag' for chain using RAG on documents provided
"""
                                  )
    chain_type: my_types.CHAIN_TYPE
    chain: RunnableSequence
    match input_chain_to_invoke:
        case 'simple':
            chain_type = my_types.CHAIN_TYPE.CHAIN_SIMPLE
            chain = chain_api.setup_chain(
                ollama_model_name=my_types.OLLAMA_MODEL_NAME
            )
        case 'rag':
            chain_type = my_types.CHAIN_TYPE.CHAIN_RAG
            chain = chain_api.setup_rag_chain(
                retriever=vectorstore.as_retriever(),
                ollama_model_name=my_types.OLLAMA_MODEL_NAME
            )
    return (chain, chain_type)
