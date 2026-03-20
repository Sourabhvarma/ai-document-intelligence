import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

load_dotenv()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# Check if DB already exists
if not os.path.exists("faiss_index"):
    print("Creating vector DB...")

    loader = PyPDFLoader("sample.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embeddings)

    # Save DB
    db.save_local("faiss_index")
    print("DB created and saved!")

else:
    print("Loading existing DB...")
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

# LLM
llm = OllamaLLM(model="llama3")

# Query loop
while True:
    query = input("\nAsk something (or type exit): ")

    if query.lower() == "exit":
        break

    results = db.similarity_search(query)

    response = llm.invoke(f"""
You are a helpful assistant.

Answer ONLY from the provided context.
If the answer is not in the context, say "Not found in document".

Context:
{results}

Question:
{query}

Answer:
""")

    print("\nAnswer:\n", response)