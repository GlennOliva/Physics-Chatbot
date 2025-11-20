import streamlit as st
import os
import asyncio

from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory   # ← FIXED
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain




# ---------------------------
# Ensure local folders exist
# ---------------------------
os.makedirs("pdfFiles", exist_ok=True)
os.makedirs("vectorDB", exist_ok=True)


# ---------------------------
# Session state variables
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Prompt template
if "prompt" not in st.session_state:
    template = """
You are a physics assistant that answers strictly based on content from the textbook PDF.

If the user asks something NOT inside the textbook, reply:
"I'm sorry, I can only provide answers based on the textbook."

Context (from textbook only):
{context}

User History:
{history}

User: {question}
Assistant (based only on textbook content):
"""
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )


# ---------------------------
# Load or create vector store
# ---------------------------
fixed_pdf_path = "phybook10.pdf"

if "vectorstore" not in st.session_state:

    if os.path.exists("vectorDB"):
        st.session_state.vectorstore = Chroma(
            persist_directory="vectorDB",
            embedding_function=OllamaEmbeddings(model="llama3.1")
        )

    else:
        # Load PDF
        loader = PyPDFLoader(fixed_pdf_path)
        documents = loader.load()

        # Split PDF into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(documents)

        st.session_state.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model="llama3.1"),
        )

        st.session_state.vectorstore.persist()


# ---------------------------
# Retriever
# ---------------------------
st.session_state.retriever = st.session_state.vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5},
)


# ---------------------------
# LLM (Ollama)
# ---------------------------
if "llm" not in st.session_state:
    st.session_state.llm = Ollama(
        base_url="http://localhost:11434",
        model="llama3.1",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


# ---------------------------
# New LCEL Retrieval Chain
# ---------------------------
if "qa_chain" not in st.session_state:
    document_chain = create_stuff_documents_chain(
        st.session_state.llm,
        st.session_state.prompt,
    )

    st.session_state.qa_chain = create_retrieval_chain(
        st.session_state.retriever,
        document_chain,
    )


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("PhyChat: A Physics Chatbot")


# Display last 5 chat messages
for msg in st.session_state.chat_history[-5:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["message"])


# ---------------------------
# Async response handling
# ---------------------------
async def get_response(user_input):

    # Show user message
    st.session_state.chat_history.append({
        "role": "user",
        "message": user_input
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("PhyChat is thinking..."):

            result = st.session_state.qa_chain.invoke({
                "input": user_input
            })

            answer = result["answer"]

        # Typing animation
        full = ""
        placeholder = st.empty()
        for ch in answer.split():
            full += ch + " "
            await asyncio.sleep(0.01)
            placeholder.markdown(full + "▌")
        placeholder.markdown(full)

    st.session_state.chat_history.append({
        "role": "assistant",
        "message": answer
    })


# ---------------------------
# Chat input
# ---------------------------
user_input = st.chat_input("Ask a question about the textbook:")

if user_input:
    asyncio.run(get_response(user_input))
