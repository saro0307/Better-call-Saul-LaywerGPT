from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os

print("Done1")
loader = DirectoryLoader('data', glob="./*.pdf", loader_cls=PyPDFLoader)
print("Done2")
documents = loader.load()

# Modify to process only the first 50 pages
for doc in documents:
    if hasattr(doc, 'content'):
        doc.content = doc.content[:50]

print("Done3")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
print("Done4")
texts = text_splitter.split_documents(documents)
print("Done5")

embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
print("Done6")
persist_directory = "ipc_vector_data"
print("Done7")
db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
print("Done8")
