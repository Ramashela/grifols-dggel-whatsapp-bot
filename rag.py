import os
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DB_PATH = "vectorstore"

def create_vectorstore():
    print("Loading PDF...")

    loader = PyPDFLoader("documents/BTS Catalogue_repaired.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(DB_PATH)

    print("Vectorstore created and saved.")


def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)


def get_qa_chain():
    if not os.path.exists(DB_PATH):
        create_vectorstore()

    vectorstore = load_vectorstore()

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4o-mini"
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )

    return qa


qa_chain = None


def get_answer(query: str) -> str:
    global qa_chain

    if qa_chain is None:
        qa_chain = get_qa_chain()

    prompt = f"""
You are a professional Grifols DG Gel assistant.

IMPORTANT RULES:
- Only answer using the provided catalogue knowledge.
- If the answer is not in the catalogue, say:
  "I’m sorry, I couldn’t find that information in the Grifols DG Gel catalogue."
- Be clear, step-by-step, and helpful for lab technicians.

User question:
{query}
"""

    response = qa_chain.run(prompt)

    return response.strip()
