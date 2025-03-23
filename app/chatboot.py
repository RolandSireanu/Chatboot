import os
import sys
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers.string import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from os.path import join, dirname
from langchain_community.document_loaders import PyPDFLoader


class ChatBoot:
    REFORMULATE_QUESTION_PROMPT = (
        "Given the chat history and latest user question,"
        "reformulate the question so it does take into account the context."
        "Very important, don't answer the question, just reformulate it if needed and otherwise return it as it is"
    )

    QUESTION_ANSWER_PROMPT = (
        "You are a question answering assistent. Use the following retrive documents to answer the question."
        "In case you don't know the answer, just say that you don't know."
        "Use three sentences and keep the answer concise"
        "\n\n"
        "{context}"
    )

    def __init__(self, workspacePath :str):
        self.current_dir = dirname(__file__)
        self.workspacePath = workspacePath
        self.persistentDirectory = os.path.join(self.workspacePath, "persistent_with_metadata")
        self.textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.chromaDB = None
        self.model = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY"))
        self.chatHistory = []

        self.reformulatePrompt = ChatPromptTemplate.from_messages([
            ("system", ChatBoot.REFORMULATE_QUESTION_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ChatBoot.QUESTION_ANSWER_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

    def ParseBook(self, bookPath:str):
        pdfLoader = PyPDFLoader(bookPath)
        pages = pdfLoader.load()
        return pages


    def addBookToVectorStore(self, filePath: str):
        pages = self.ParseBook(filePath)
        splitedDocs = self.textSplitter.split_documents(pages)

        if not os.path.exists(self.persistentDirectory):
            print("Chroma db doesn't exists, creating it ...")
            self.chromaDB = Chroma.from_documents(splitedDocs,
                embedding=OpenAIEmbeddings(model="text-embedding-3-small"), 
                persist_directory=self.persistentDirectory)
            
            print("\n--- Finished creating vector store ---\n")
        else:
            self.chromaDB = Chroma(persist_directory=self.persistentDirectory, embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
            print("Vector store already exists, no need to recreate it")
            self.chromaDB.add_documents(splitedDocs)

    def query(self, question :str) -> str:
        if (self.chromaDB is not None):
            chromaDBRetriver = self.chromaDB.as_retriever(search_type="similarity", 
                                            search_kwargs = {"k":3})
            history_aware_retriver = create_history_aware_retriever(llm=self.model, retriever=chromaDBRetriver, prompt = self.reformulatePrompt)
            qa_chain = create_stuff_documents_chain(llm=self.model, prompt=self.qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriver, qa_chain)

            response = rag_chain.invoke({"input": question, "chat_history": self.chatHistory})
            print(f"AI : {response["answer"]}")

            self.chatHistory.append(HumanMessage(question))
            self.chatHistory.append(SystemMessage(response["answer"]))

            return response["answer"]
        else:
            return "No pdf file uploaded yet"
