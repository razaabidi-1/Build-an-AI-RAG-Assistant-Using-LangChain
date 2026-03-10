"""
run_tasks.py  –  Run all 6 RAG tasks and print output for screenshots.
Usage:  python run_tasks.py
"""
import os, sys
os.chdir('/workspaces/Build-an-AI-RAG-Assistant-Using-LangChain')
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN', '')

# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 – Load Document Using LangChain   →  screenshot: pdf_loader.png
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  TASK 1 – Load Document Using LangChain (PDF Loader)")
print("  📸 Screenshot this output → save as  pdf_loader.png")
print("═"*62)

from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "sample_paper.pdf"
loader = PyPDFLoader(PDF_PATH)
pages  = loader.load()

print(f"\nfrom langchain_community.document_loaders import PyPDFLoader\n")
print(f"PDF_PATH = 'sample_paper.pdf'")
print(f"loader   = PyPDFLoader(PDF_PATH)")
print(f"pages    = loader.load()\n")
print(f"✅  PDF loaded successfully")
print(f"   Total pages  : {len(pages)}")
print(f"   Document type: {type(pages[0]).__name__}")
print(f"\n── First page metadata ──")
print(pages[0].metadata)
print(f"\n── First 400 chars of page 1 ──")
print(pages[0].page_content[:400])

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 – Apply Text Splitting Techniques  →  screenshot: code_splitter.png
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "═"*62)
print("  TASK 2 – Apply Text Splitting Techniques")
print("  📸 Screenshot this output → save as  code_splitter.png")
print("═"*62)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size      = 1000,
    chunk_overlap   = 200,
    length_function = len,
    separators      = ["\n\n", "\n", " ", ""]
)

splits = text_splitter.split_documents(pages)

print(f"""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size      = 1000,
    chunk_overlap   = 200,
    length_function = len,
    separators      = ["\\n\\n", "\\n", " ", ""]
)

splits = text_splitter.split_documents(pages)
""")
print(f"✅  Text splitting complete")
print(f"   Original pages  : {len(pages)}")
print(f"   Total chunks    : {len(splits)}")
print(f"   Chunk size      : 1000 chars  |  Overlap: 200 chars")
print(f"\n── Sample chunk (index 2) ──")
print(f"Content ({len(splits[2].page_content)} chars):")
print(splits[2].page_content)
print(f"\nMetadata: {splits[2].metadata}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 – Embed Documents   →  screenshot: embedding.png
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "═"*62)
print("  TASK 3 – Embed Documents (HuggingFace Embeddings)")
print("  📸 Screenshot this output → save as  embedding.png")
print("═"*62)

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name    = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs  = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True},
)

sample_text   = splits[0].page_content
sample_vector = embeddings.embed_query(sample_text)

print(f"""
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name    = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs  = {{"device": "cpu"}},
    encode_kwargs = {{"normalize_embeddings": True}},
)
""")
print(f"✅  Embedding model loaded: sentence-transformers/all-MiniLM-L6-v2")
print(f"   Runs locally on CPU – FREE, no API key required")
print(f"   Embedding dimensions : {len(sample_vector)}")
print(f"   Sample text (first 100 chars): {sample_text[:100]}...")
print(f"\n   First 10 embedding values:")
print([round(v, 6) for v in sample_vector[:10]])

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 – Create Chroma Vector Database  →  screenshot: vectordb.png
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "═"*62)
print("  TASK 4 – Create Chroma Vector Database")
print("  📸 Screenshot this output → save as  vectordb.png")
print("═"*62)

from langchain_community.vectorstores import Chroma

PERSIST_DIR = "./chroma_db"

vectordb = Chroma.from_documents(
    documents         = splits,
    embedding         = embeddings,
    persist_directory = PERSIST_DIR,
)

print(f"""
from langchain_community.vectorstores import Chroma

PERSIST_DIR = "./chroma_db"

vectordb = Chroma.from_documents(
    documents         = splits,
    embedding         = embeddings,
    persist_directory = PERSIST_DIR,
)
""")
print(f"✅  Chroma vector database created")
print(f"   Persist directory  : {PERSIST_DIR}")
print(f"   Documents indexed  : {vectordb._collection.count()}")

test_query = "What is the main topic of this paper?"
results    = vectordb.similarity_search(test_query, k=2)
print(f"\n── Similarity search test ──")
print(f"Query: '{test_query}'")
for i, doc in enumerate(results):
    print(f"\nResult {i+1} (page {doc.metadata.get('page', 'N/A')}):")
    print(doc.page_content[:250])

# ══════════════════════════════════════════════════════════════════════════════
# TASK 5 – Retriever   →  screenshot: retriever.png
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "═"*62)
print("  TASK 5 – Develop a Retriever")
print("  📸 Screenshot this output → save as  retriever.png")
print("═"*62)

retriever = vectordb.as_retriever(
    search_type   = "similarity",
    search_kwargs = {"k": 3}
)

query = "What this paper is talking about?"
docs  = retriever.invoke(query)

print(f"""
retriever = vectordb.as_retriever(
    search_type   = "similarity",
    search_kwargs = {{"k": 3}}
)

query = "What this paper is talking about?"
docs  = retriever.invoke(query)
""")
print(f"✅  Retriever configured (ChromaDB | similarity | top-3)")
print(f"   Query            : {query}")
print(f"   Chunks retrieved : {len(docs)}")
print()
for i, doc in enumerate(docs):
    print(f"{'─'*55}")
    print(f"Chunk {i+1}  |  Source page: {doc.metadata.get('page', 'N/A')}")
    print(doc.page_content[:350])
    print()

print("\n" + "═"*62)
print("  ✅  Tasks 1–5 complete!")
print("  Next: add your HF token to .env, then run app.py for Task 6")
print("═"*62)
