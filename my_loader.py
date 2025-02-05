from langchain_core.document_loaders.base import BaseLoader

class MyLoader:
    loader: BaseLoader
    name: str
    def __init__(self, loader: BaseLoader, name: str= "BaseLoader"):
        self.loader = loader
        self.name = name

    def get_loader(self) -> BaseLoader:
        return self.loader

    def get_description(self) -> str:
        return f"{self.name}"