import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY in .env file")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAI
from langchain.chains import RetrievalQA

# Path to your PDF
PDF_PATH = Path("documents/BTS Catalogue_repaired.pdf")
VECTORSTORE_PATH = Path("vectorstore/faiss_index")

# Ensure vectorstore folder exists
VECTORSTORE_PATH.parent.mkdir(exist_ok=True, parents=True)

# Load PDF and create vector store
def create_vectorstore():
    if not PDF_PATH.exists():
        print(f"⚠️ PDF file not found at {PDF_PATH}. Bot will return default responses.")
        return None

    print(f"📄 Loading PDF: {PDF_PATH}")
    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAI(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save vectorstore for future use
    FAISS.save_local(vectorstore, str(VECTORSTORE_PATH))
    print(f"✅ Vectorstore created and saved at {VECTORSTORE_PATH}")
    return vectorstore

# Load vectorstore if exists, else create
if VECTORSTORE_PATH.exists():
    print("📦 Loading existing vectorstore...")
    vectorstore = FAISS.load_local(str(VECTORSTORE_PATH), OpenAI(openai_api_key=OPENAI_API_KEY))
else:
    vectorstore = create_vectorstore()

# Setup RetrievalQA chain
if vectorstore:
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=OPENAI_API_KEY),
        retriever=vectorstore.as_retriever()
    )
else:
    qa = None  # Will handle gracefully in get_answer

# Function to get answer from PDF via AI
def get_answer(query: str) -> str:
    if not vectorstore or not qa:
        return "⚠️ PDF not loaded yet. Please upload the PDF in documents/BTS Catalogue_repaired.pdf"

    try:
        response = qa.run(query)
        if not response:
            return "❌ No answer found in the catalogue for your query."
        return response
    except Exception as e:
        print("[ERROR in get_answer]", e)
        return "⚠️ Something went wrong while processing your query."
