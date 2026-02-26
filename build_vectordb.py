from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

print("Loading documents...")

# Load all text files
loader = DirectoryLoader(
    './science_docs/', 
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)

documents = loader.load()
print(f"Loaded {len(documents)} documents")

# Split documents into chunks
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Create embeddings
print("Creating embeddings (this may take 2-3 minutes)...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Create vector database
print("Building vector database...")
vectordb = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)
vectordb.save_local("./faiss_index")

print("Vector database created successfully!")
print(f"Location: ./faiss_index")
print(f"Total chunks indexed: {len(chunks)}")