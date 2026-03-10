"""
run_tasks.py  –  Quest Analytics AI RAG Assistant
Prints labelled output for all 6 assignment tasks.
Usage:  .venv/bin/python3 run_tasks.py
"""
import os, warnings
warnings.filterwarnings("ignore")
os.chdir('/workspaces/Build-an-AI-RAG-Assistant-Using-LangChain')
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN', '')

W = 64  # banner width

def banner(task_num, title, caption, filename):
    print("\n" + "█"*W)
    print(f"  📌 TASK {task_num}  –  {title}")
    print(f"  📝 {caption}")
    print(f"  📸 Save screenshot as:  {filename}")
    print(f"  🏢 Quest Analytics  |  AI RAG Assistant  |  LangChain")
    print("█"*W + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 – Load Document Using LangChain
# ══════════════════════════════════════════════════════════════════════════════
banner(1,
       "Load Document Using LangChain (PDF Loader)",
       "Loading a PDF with PyPDFLoader – each page becomes a Document object",
       "pdf_loader.png")

from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "sample_paper.pdf"
loader = PyPDFLoader(PDF_PATH)
pages  = loader.load()

print("─── Code ─────────────────────────────────────────────────────")
print("from langchain_community.document_loaders import PyPDFLoader\n")
print("PDF_PATH = 'sample_paper.pdf'")
print("loader   = PyPDFLoader(PDF_PATH)")
print("pages    = loader.load()")
print("─── Output ───────────────────────────────────────────────────")
print(f"✅  PDF loaded successfully")
print(f"   Total pages  : {len(pages)}")
print(f"   Document type: {type(pages[0]).__name__}")
print(f"\n── First page metadata ──")
print(pages[0].metadata)
print(f"\n── First 400 chars of page 1 ──")
print(pages[0].page_content[:400])

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 – Apply Text Splitting Techniques
# ══════════════════════════════════════════════════════════════════════════════
banner(2,
       "Apply Text Splitting Techniques",
       "tSplitting pages into 1000-char chunks (overlap=200) for LLM contex",
       "code_splitter.png")

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size      = 1000,
    chunk_overlap   = 200,
    length_function = len,
    separators      = ["\n\n", "\n", " ", ""]
)

splits = text_splitter.split_documents(pages)

print("─── Code ─────────────────────────────────────────────────────")
print(f"""text_splitter = RecursiveCharacterTextSplitter(
    chunk_size      = 1000,
    chunk_overlap   = 200,
    length_function = len,
    separators      = ["\\n\\n", "\\n", " ", ""]
)
splits = text_splitter.split_documents(pages)""")
print("─── Output ───────────────────────────────────────────────────")
print(f"✅  Text splitting complete")
print(f"   Original pages  : {len(pages)}")
print(f"   Total chunks    : {len(splits)}")
print(f"   Chunk size      : 1000 chars  |  Overlap: 200 chars")
print(f"\n── Sample chunk (index 2) ──")
print(f"Content ({len(splits[2].page_content)} chars):")
print(splits[2].page_content[:400])
print(f"\nMetadata: {splits[2].metadata}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 – Embed Documents
# ══════════════════════════════════════════════════════════════════════════════
banner(3,
       "Embed Documents Using HuggingFace Embeddings",
       "Converting text chunks into 384-dim vectors using sentence-transformers (local CPU)",
       "embedding.png")

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name    = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs  = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True},
)

sample_text   = splits[0].page_content
sample_vector = embeddings.embed_query(sample_text)

print("─── Code ─────────────────────────────────────────────────────")
print("""from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name    = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs  = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True},
)""")
print("─── Output ───────────────────────────────────────────────────")
print(f"✅  Embedding model loaded: sentence-transformers/all-MiniLM-L6-v2")
print(f"   Runs locally on CPU – FREE, no API key required")
print(f"   Embedding dimensions : {len(sample_vector)}")
print(f"\n   First 10 embedding values:")
print([round(v, 6) for v in sample_vector[:10]])

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 – Create Chroma Vector Database
# ══════════════════════════════════════════════════════════════════════════════
banner(4,
       "Create and Configure Chroma Vector Database",
       "Indexing all embedded chunks in ChromaDB for fast similarity retrieval",
       "vectordb.png")

from langchain_community.vectorstores import Chroma

PERSIST_DIR = "./chroma_db"

vectordb = Chroma.from_documents(
    documents         = splits,
    embedding         = embeddings,
    persist_directory = PERSIST_DIR,
)

print("─── Code ─────────────────────────────────────────────────────")
print("""from langchain_community.vectorstores import Chroma

PERSIST_DIR = "./chroma_db"
vectordb = Chroma.from_documents(
    documents         = splits,
    embedding         = embeddings,
    persist_directory = PERSIST_DIR,
)""")
print("─── Output ───────────────────────────────────────────────────")
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
# TASK 5 – Develop a Retriever
# ══════════════════════════════════════════════════════════════════════════════
banner(5,
       "Develop a Retriever to Fetch Relevant Document Segments",
       "Configuring ChromaDB as a LangChain Retriever (similarity, top-3 chunks)",
       "retriever.png")

retriever = vectordb.as_retriever(
    search_type   = "similarity",
    search_kwargs = {"k": 3}
)

query = "What this paper is talking about?"
docs  = retriever.invoke(query)

print("─── Code ─────────────────────────────────────────────────────")
print("""retriever = vectordb.as_retriever(
    search_type   = "similarity",
    search_kwargs = {"k": 3}
)
query = "What this paper is talking about?"
docs  = retriever.invoke(query)""")
print("─── Output ───────────────────────────────────────────────────")
print(f"✅  Retriever configured (ChromaDB | similarity | top-3)")
print(f"   Query            : {query}")
print(f"   Chunks retrieved : {len(docs)}")
print()
for i, doc in enumerate(docs):
    print(f"{'─'*55}")
    print(f"Chunk {i+1}  |  Source page: {doc.metadata.get('page', 'N/A')}")
    print(doc.page_content[:350])
    print()

# ══════════════════════════════════════════════════════════════════════════════
# TASK 6 – QA Bot (Gradio UI)
# ══════════════════════════════════════════════════════════════════════════════
banner(6,
       "Construct a QA Bot Using LangChain + LLM + Gradio",
    "Gradio web UI wiring PyPDFLoader, ChromaDB, RetrievalQA & a local FLAN-T5 LLM",
       "QA_bot.png")

print("  ── How to get the Task 6 screenshot ─────────────────────── ")
print()
print("  1. Open a NEW terminal and run:")
print("       .venv/bin/python3 app.py")
print()
print("  2. Wait for the Gradio URL to appear:") 
print("       Running on public URL: https://xxxx.gradio.live")
print()
print("  3. Open that URL in your browser, then:")
print("       ① Click  Upload PDF  →  select  sample_paper.pdf")
print("       ② Click  📤 Index PDF  →  wait for ✅ confirmation")
print("       ③ Type:  What this paper is talking about?")
print("       ④ Click  Send ➤  →  wait for the answer")
print()
print("  4. Take a full-browser screenshot → save as  QA_bot.png")
print()
print("─"*W)
print()

print("█"*W)
print("  ✅  ALL TASKS COMPLETE  –  Quest Analytics RAG Assistant")
print(f"  Tasks 1–5: take screenshots from the output above")
print(f"  Task 6   : run  .venv/bin/python3 app.py  for the bot UI")
print("█"*W + "\n")
