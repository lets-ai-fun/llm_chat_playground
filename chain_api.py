from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.language_models.base import BaseLanguageModel
from langchain_ollama import ChatOllama
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.runnables.utils import Output
from langchain.schema.output_parser import StrOutputParser

from my_types import CHAIN_TYPE


def setup_chain(
        ollama_model_name: str) -> RunnableSequence:
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            SystemMessage(
                content="You are a helpful assistant."
            ),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )

    llm = ChatOllama(model=ollama_model_name)
    chain = prompt | llm
    return chain


def setup_rag_chain(
        retriever: RetrieverLike,
        ollama_model_name: str) -> RunnableSequence:
    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(model=ollama_model_name)
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
    )

    return chain


def endless_chat(
        chain: RunnableSequence,
        chain_type: CHAIN_TYPE) -> None:
    """
    Run a chain endlessy.
    Special input:
    - quit : end chain execution
    - disable_chain_log : disable chain callback logging
    - enable_chain_log : enable chain callback logging
    :param chain: chain
    :return: none
    """

    while True:
        user_message = input("input message ['quit' to exit]:")
        match user_message:
            case 'quit':
                break

        chain_input: any
        match chain_type:
            case CHAIN_TYPE.CHAIN_RAG: chain_input = user_message
            case CHAIN_TYPE.CHAIN_SIMPLE: chain_input = {"input": user_message}

        result: Output = chain.invoke(
            chain_input
        )

        print(f"{result.content}")
