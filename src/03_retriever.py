from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

load_dotenv()

# Load all .txt files from the data folder using plain Python
data_folder = "data"
documents = []
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append(Document(page_content=text, metadata={"source": filename}))

# Split documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print(f"Loaded {len(documents)} documents, split into {len(chunks)} chunks.")

# Embed and store in Chroma (persisted to disk in ./chroma_db)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector store built and saved.")

# Test query
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

if __name__ == "__main__":
    query = "Is damage to my car in a parking lot covered?"
    results = retriever.invoke(query)
    print(f"\nQUERY: {query}\n")
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} (from {doc.metadata.get('source')}) ---")
        print(doc.page_content)
        print()