from langchain_classic.schema import BaseChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import os
from langchain.chains import create_history_aware_retriever, create_retriever_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_history import ChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbedding
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


st.title("Conversation with History RAG with uploaded PDF")
st.write("Upload a PDF to get started")

api_key=st.text_input("Groq API Key")
os.environ["GROQ_API_KEY"] = api_key



if api_key:
    llm=ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
    session_id = st.text_input("Session ID", value="default")
    
    # Manage conversation history
    if 'store' not in st.session_state:
        st.session_state.store = {

        }
    
    uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True)
    if uploaded_files:
        documents = []
        for file in uploaded_files:
            temp_pdf =f"./temp.pdf"
            with open(temp_pdf, "wb") as f:
                f.write(file.getvalue())
                file_name = file.name

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)
            os.remove(temp_pdf)

        # Split and create embeddings for documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500,
            separators=["\n\n", "\n", " ", ""],
        )
        split_docs = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()

    contextulize_q_system_prompt=(
        "Give a chat history and latest question"
        "which may reference the chat history,"
        "formulate a standalone question which can be understood"
        "without the chat history. DO NOT ANSWER THE QUESTION,"
        "just reformulate it if needed and otherwise return it as is."
    )
    contextulize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextulize_q_system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm,retriever,contextulize_q_prompt)
    

    ##Answer the question prompt
    system_prompt = (
        "You are an assistant for question-answering tasks."
        "Use the following pieces of retriver context to answer"
        "the question. If you don't know the answer, say that you don't know."
        "Use three sentences maximum and keep the answer as concise as possible."
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    ])


    question_answer_chain = create_stuff_documents_chain(
        llm, qa_prompt
    )

    rag_chain = create_retriever_chain(
        history_aware_retriever,
        question_answer_chain
    )


    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,
        get_session_history(session_id),
        input_message_key="input",
        history_message_key="chathistory",
        output_message_key="answer"
    )

    user_input = st.text_input("User Input", key="user_input")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
            )