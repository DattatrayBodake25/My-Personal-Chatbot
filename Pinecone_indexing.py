# pip install langchain openai sentence_transformers unstructured detectron2 pinecone-client
# !pip install unstructured[local-inference] -q
# !pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6


import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone
from langchain.vectorstores import Pinecone

# List of file paths to the PDFs
pdf_paths = [
    r"C:\Users\bodak\OneDrive\Desktop\Chat Task\02-Biggby Coffee April 30, 2021 FDD-clean-final v2.pdf",
    r"C:\Users\bodak\OneDrive\Desktop\Chat Task\08-21 Wahlburgers Tribal Casino FDD (Corrected 080521) (1).PDF",
    r"C:\Users\bodak\OneDrive\Desktop\Chat Task\2019 Bloomin Blinds FDD.pdf",
    r"C:\Users\bodak\OneDrive\Desktop\Chat Task\2021 Amazing Athletes FDD.ISSUED.4.30.21.pdf",
    r"C:\Users\bodak\OneDrive\Desktop\Chat Task\Atomic Wings AR FDD-Complete Clean.pdf",
]

# Function to load documents from file paths
def load_docs(paths):
    documents = []
    for path in paths:
        with fitz.open(path) as pdf_document:
            text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text()

            document = {"content": text, "metadata": {"source": path}}
            documents.append(document)
    return documents

# Load documents from the PDFs
documents = load_docs(pdf_paths)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
texts, metadatas = [], []
for doc in documents:
    texts.append(doc["content"])
    metadatas.append(doc["metadata"])
docs = text_splitter.create_documents(texts, metadatas=metadatas)

# Initialize Sentence Transformer for embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Pinecone
pinecone.init(api_key="my_pinecone_api_key", environment="my_pinecone_env")

# Create an index in Pinecone
index_name = "my_index_name"
try:
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    print("Index creation successful.")
except Exception as e:
    print(f"Error during index creation: {e}")

# Function to get similar documents based on user query
def get_similar_docs(query, k=1, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

# User query
user_query = "What are the main offerings at BIGGBY® COFFEE?"

# Get similar documents based on the user query
similar_docs = get_similar_docs(user_query)
print("Similar Documents:")
print(similar_docs)