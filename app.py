import streamlit as st
import os
import asyncio

from langchain_core.prompts import PromptTemplate
from langchain_community.memory import ConversationBufferMemory

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# ---------------------------
# Ensure folders exist
# ---------------------------
os.makedirs("pdfFiles", exist_ok=True)
os.makedirs("vectorDB", exist_ok=True)


# ---------------------------
# API Key
# ---------------------------
GEMINI_API_KEY = st.secrets["AIzaSyCmJMX_NxAeOsAtXPsZVaePJysKTiFROdY"]


# ---------------------------
# Session state
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "prompt" not in st.session_state:
    template = """
You are a physics assistant that answers strictly from the textbook PDF.

If the user asks something NOT inside the textbook, reply:
"I'm sorry, I can only provide answers based on the textbook."

Context (from textbook):
{context}

Conversation history:
{history}

User: {question}
Assistant:
"""
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template
    )


# ---------------------------
# Load PDF + Vector Store
# ---------------------------
fixed_pdf_path = "phybook10.pdf"

if "vectorstore" not in st.session_state:

    loader = PyPDFLoader(fixed_pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)

    # --- Cloud Embeddings (Gemini) ---
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    st.session_state.vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="vectorDB"
    )

    st.session_state.vectorstore.persist()


# ---------------------------
# Retriever
# ---------------------------
st.session_state.retriever = st.session_state.vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)


# ---------------------------
# LLM (Gemini)
# ---------------------------
if "llm" not in st.session_state:
    st.session_state.llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )


# ---------------------------
# Retrieval Chain
# ---------------------------
if "qa_chain" not in st.session_state:
    document_chain = create_stuff_documents_chain(
        st.session_state.llm,
        st.session_state.prompt
    )

    st.session_state.qa_chain = create_retrieval_chain(
        st.session_state.retriever,
        document_chain
    )


# ---------------------------
# UI
# ---------------------------
st.title("PhyChat – Physics Chatbot (PDF-Based)")


for msg in st.session_state.chat_history[-5:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["message"])


# ---------------------------
# Async Response
# ---------------------------
async def get_response(user_input):

    st.session_state.chat_history.append({"role": "user", "message": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            result = st.session_state.qa_chain.invoke({"input": user_input})
            answer = result["answer"]

        # fake typing effect
        placeholder = st.empty()
        full = ""
        for w in answer.split():
            full += w + " "
            await asyncio.sleep(0.01)
            placeholder.markdown(full + "▌")

        placeholder.markdown(full)

    st.session_state.chat_history.append({"role": "assistant", "message": answer})


# ---------------------------
# Input
# ---------------------------
user_input = st.chat_input("Ask something from the textbook...")

if user_input:
    asyncio.run(get_response(user_input))
