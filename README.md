# LLM Chat Playground
A basic python llm project to use as playground in implementing different LangChain application types.
  
The purpose of the project is to provide organized code that allows experimentation with different forms of interaction with an LLM.

At the current stage, the following functionalities are implemented:
 
- Creation of a simple chain (i.e. input passed directly to the LLM)
- Creation of a chain with RAG support (using Chroma as a vector store)
- Use of Ollama as the LLM and Embedding provider
  
**More chain type support (different RAG type), LangGraph graph and features will be added soon!**

# Prerequisites
- Install [Ollama](https://ollama.com)
# To run:
## Ollama Setup
- pull models (update the models to download if you change the model you use in program): you can perfrom this just once as model are cached on your local machine once dowloaded
  - `ollama pull llama3.2`
  - `ollama pull nomic-embed-text`
- run ollama `ollama serve`

## Dependencies Setup
`pip install -r requirements.txt`
## Document for RAG Setup
- Place PDFs to be used for RAG in the `documents` folder.
## Run 
- run using `python main.py`
  - Note: I use Python 3.12.0 

# Program Flow
## 1 - Vector store creation
At the start, you will be prompted whether you want to clear the vector store:  
- **If you respond with 'y'**, the vector store will be deleted, and the documents in the `documents` folder will be loaded, chunked, and their corresponding embedded vectors will be saved in the vector store.  
- **If you respond with 'n'**, the vector store will not be updated, but its existing content will be loaded.  
**Note:**  
On the first run, the vector store does not exist (as it is not included in the GitHub repository). If you choose not to create it, you will simply have an empty vector store. In this case, using the chain with RAG will likely result in irrelevant or no responses.
### Vector store loader
A couple of loaders are used
- one `WebBaseLoader` to scrape content from url 'https://lilianweng.github.io/posts/2023-06-23-agent/'
- n `document loader`, one for each docuemnt in 'documents' folder
To update this behaviour please look at `main.py` in `setup_vectorstore` method where you can find comments explaining the needed code changes.
## 2 - Chain execution
After setting up the vector store, you will be prompted to choose the type of chain to use:

- **Enter `simple`** for a chain without RAG, where the input is passed directly to the LLM, and the generated response is displayed.  
- **Enter `rag`** for a RAG-based chain, where retrieved data from the vector store (based on the user input) is used as context.  
- **Enter `quit`** to exit the program.

# Details
The project is organized into classes with specific purposes:
- `main.py` serves as the entry point
- `chain_api.py` handles the creation of the chain and the execution loop (once the chain is created, users can ask questions until they enter 'quit' to exit the program)
- `doc_loader_api.py` provides loaders for loading documents from the filesystem
- `vectorstore_api.py`: Creates a collection in the vector store from the documents loaded by the loader.  
- `my_loader.py`: A wrapper around a `BaseLoader` to include additional information (currently, a description useful for debugging and identifying the loader being used).  
- `my_types.py`: A class containing enums to better define the type of chain being used (either a simple chain or an RAG chain).
