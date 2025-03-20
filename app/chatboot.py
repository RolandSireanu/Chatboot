import dotenv
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

current_dir = dirname(__file__)
pdfPath = os.path.join(current_dir, "resources")
persistentDirectory = os.path.join(current_dir, "persistent_with_metadata")

model = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY"))

def ParseBook(bookPath:str):
    pdfLoader = PyPDFLoader(bookPath)
    docs = pdfLoader.load()
    return docs

chromaDB = None

if not os.path.exists(persistentDirectory):
    print("Chroma db doesn't exists, creating it ...")

    if(pdfPath == ""):
        raise FileNotFoundError(f"The directory {pdfPath} doesn't exist")
    
    books = [book for book in os.listdir(pdfPath) if book.endswith(".pdf")]
    docs = []
    for b in books:
        for d in ParseBook(os.path.join(pdfPath,b)):
            d.metadata = {"source": b}
            docs.append(d)

    textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splitedDocs = textSplitter.split_documents(docs)

    chromaDB = Chroma.from_documents(docs, 
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"), 
        persist_directory=persistentDirectory)
    
    print("\n--- Finished creating cvector store ---\n")
else:
    chromaDB = Chroma(persist_directory=persistentDirectory, embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
    print("Vector store already exists, no need to recreate it")
 
chromaDBRetriver = chromaDB.as_retriever(search_type="similarity", 
                                        search_kwargs = {"k":3})

reformulateQuestion=(
    "Given the chat history and latest user question,"
    "reformulate the question so it does take into account the context."
    "Very important, don't answer the question, just reformulate it if needed and otherwise return it as it is"
)

reformulatePrompt = ChatPromptTemplate.from_messages([
    ("system", reformulateQuestion),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

question_answer_prompt = (
    "You are a question answering assistent. Use the following retrive documents to answer the question."
    "In case you don't know the answer, just say that you don't know."
    "Use three sentences and keep the answer concise"
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", question_answer_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)


history_aware_retriver = create_history_aware_retriever(llm=model, retriever=chromaDBRetriver, prompt = reformulatePrompt)
qa_chain = create_stuff_documents_chain(llm=model, prompt=qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriver, qa_chain)


def chat():
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        response = rag_chain.invoke({"input": query, "chat_history": chat_history})
        print(f"AI : {response["answer"]}")

        chat_history.append(HumanMessage(query))
        chat_history.append(SystemMessage(response["answer"]))

chat()