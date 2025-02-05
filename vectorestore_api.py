from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from typing import List
import os
import chromadb as chromadb_direct

from my_loader import MyLoader

__collection_name__ = "documents"


def collection_name() -> str:
    """
    Vector store collection name containing embedded vectors
    :return: collection name containing embedded vectors
    """
    return __collection_name__


def setup_vectorstore(
        loader: [MyLoader],
        db_folder: str,
        embeddings: Embeddings,
        chunk_size: int,
        chunk_overlap: int) -> VectorStore:
    """
    setup vectorstore by loading docuemnt(s) using a list of loaders.
    The vectorestore is persisted in specified folder
    :param loader: loader(s) to load documents
    :param db_folder: folder to save vectorestore db with saved embeddings from laoded document(s)
    :param embeddings: embeddings to use while persisting data
    :param chunk_size: document chunck size
    :param chunk_overlap: document chunk overlap
    :return:
    """
    docs = __load_documents(loader)
    all_splits = __chuck_documents(chunk_overlap, chunk_size, docs)
    if len(all_splits) == 0:
        print(f"no document found to populate vectorstore")
        raise SystemExit(f"no document found to populate vectorstore")

    print(f"deleting collection '{__collection_name__}' from vectorstore")
    #Chroma(persist_directory= db_folder).delete_collection(name= __collection_name__)

    # langchain Chroma client doesn't allow to remove specific collection before cerating one in the same code
    # DEtails: the attribute collection.name is setup in ' Chroma.from_documents(....)' method:
    # before this method invocation it's set to the constant _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain" .
    #
    # SO, to delete the specific colelction, we've to resort to chromadb python package
    chromaClient = chromadb_direct.PersistentClient(path= db_folder)
    chromaClient.delete_collection(name= __collection_name__)
    print(f"loading {len(all_splits)} chunks into vectorstore collection '{__collection_name__}'...please wait")
    vector_store = Chroma.from_documents(documents= all_splits,
                                         collection_name= __collection_name__,
                                         embedding= embeddings,
                                         persist_directory= db_folder)
    print(f"vector store #chunks {len(vector_store.get()[__collection_name__])}")
    return vector_store


def load_vectorstore(
        db_folder: str,
        embeddings: Embeddings) -> VectorStore:
    """
    Initialize vectorestore using persisted data
    :param db_folder: folder containing persisted data
    :param embeddings: embeddings used to persist data
    :return:
    """
    print(f"reuse db in folder {db_folder}")
    vector_store = __load_vectorstore(db_folder= db_folder, embeddings= embeddings)
    print(f"vector store #chunks {len(vector_store.get()[__collection_name__])}")
    return vector_store


def __get_stored_documents(vectorstore: VectorStore, collection: str = __collection_name__) -> [str]:
    """
    Retrieve vector store document stored in collection
    :param vectorstore: vectorstore to get documents from
    :return: documents contained in vectorstore
    """
    return vectorstore.get()[collection]


def __chuck_documents(
        chunk_overlap: int,
        chunk_size: int,
        docs: [Document]) -> List[Document]:
    """
    Create chunks from documents using RecursiveCharacterTextSplitter
    :param chunk_overlap: chunk overlap param for RecursiveCharacterTextSplitter
    :param chunk_size: chunk size param for RecursiveCharacterTextSplitter
    :param docs: documents to extract chunk from
    :return:
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(docs)
    return all_splits


def __load_documents(loaders: [MyLoader]) -> List[Document]:
    """
    load documents using provided loader
    :param loader:
    :return:
    """
    docs = [__load_document_debug(loader) for loader in
            loaders]  # each item in docs is a list of document returned from a loader
    return sum(docs, [])  # flatten docs


def __load_document_debug(loader: MyLoader) -> [Document]:
    """
    wrapper to perform docuemnt loading while printng debug info
    :param loader: loader with description
    :return: loaded document(s)
    """
    print(f"loading document(s) using loader '{loader.get_description()}'")
    return loader.get_loader().load()


def __load_vectorstore(
        db_folder: str,
        embeddings: Embeddings) -> VectorStore:
    """
    Load vectorstore data from folder previously populated
    :param db_folder: db folder
    :param embeddings: embeddings
    :return: vectorstore initialized with data from db folder
    """
    if False == os.path.exists(db_folder):
        raise SystemExit(f"vector store folder [{db_folder}] not found")

    return Chroma(collection_name=__collection_name__,
                  persist_directory=db_folder,
                  embedding_function=embeddings)
