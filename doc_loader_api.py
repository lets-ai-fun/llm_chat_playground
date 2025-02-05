from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import fnmatch
import os

from my_loader import MyLoader

def get_loaders(doc_folder: str,
              suffix: str) -> list[MyLoader]:
    """
    load document in doc_folder
    :param doc_folder: document folder
    :return: langchain Document list
    """
    print(f"looking fo {suffix} files in folder {doc_folder}")
    files = __filter_file(doc_folder, suffix)
    loaders= [MyLoader(loader= PyPDFLoader(file), name= file) for file in
             files]  # TODO improve memory usage when large number of docs
    return loaders


def __filter_file(folder: str,
                  suffix: str) -> list[str]:
    """
    Filter file in folder by checking suffix.
    :param folder: file(s) folder
    :param suffix: file suffix to check
    :return: file(s) in folder with suffix
    """
    files= [file for file in [os.path.join(folder, file) for file in os.listdir(folder)]
            if fnmatch.fnmatch(name=file, pat="*" + suffix) and os.path.isfile(file)
            ]
    print(f"file(s) found:{files}")
    return files


def __check_document_availabile(doc_folder: str,
                                suffix: str) -> bool:
    """
    Check if doc_folder contains at least one document with suffix
    :param doc_folder: document folder
    :param suffix: document suffix
    :return: true if at least one document with suffix found, false otherwise
    """
    files= [file for file in [os.path.join(doc_folder, file) for file in os.listdir(doc_folder)] # TODO improve logic to avoid retrieving doc list from folder more than once
            if fnmatch.fnmatch(name=file, pat="*" + suffix) and os.path.isfile(file)
            ]
    return len(files) > 0

