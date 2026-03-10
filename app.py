"""
app.py – Standalone Gradio QA Bot
Quest Analytics – AI RAG Assistant  (Free – local HuggingFace models)

Run:
        python app.py

Then open the printed URL in your browser.
"""

import os, tempfile
from dotenv import load_dotenv

load_dotenv()

# ─── Optional credentials ─────────────────────────────────────────────────────
# Kept for compatibility, but the current QA bot runs fully locally.
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# ─── Imports ──────────────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline
import gradio as gr

# ─── Text Splitter (shared) ───────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 1000,
    chunk_overlap = 200,
    length_function = len,
    separators    = ["\n\n", "\n", " ", ""],
)

# ─── Embedding Model (runs 100% locally – no API key needed) ─────────────────
embeddings = HuggingFaceEmbeddings(
    model_name    = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs  = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True},
)

# ─── LLM (free local Transformers pipeline) ──────────────────────────────────
generation_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_new_tokens=256,
    temperature=0.3,
)

llm = HuggingFacePipeline(pipeline=generation_pipeline)

# ─── Prompt Template ──────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are a helpful research assistant for Quest Analytics.
Use ONLY the following context excerpts from the document to answer the question.
If the answer is not contained in the context, say "I don't have enough information in the provided document to answer that."

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE,
)

# ─── Shared State ─────────────────────────────────────────────────────────────
current_chain: dict = {"chain": None}


def build_qa_chain(pdf_path: str) -> RetrievalQA:
    """Load PDF → split → embed → Chroma → RetrievalQA."""
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    splits = text_splitter.split_documents(pages)
    vdb    = Chroma.from_documents(splits, embedding=embeddings)
    ret    = vdb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    chain  = RetrievalQA.from_chain_type(
        llm                     = llm,
        chain_type              = "stuff",
        retriever               = ret,
        return_source_documents = True,
        chain_type_kwargs       = {"prompt": prompt},
    )
    return chain


def upload_pdf(file) -> str:
    """Gradio callback: index the uploaded PDF."""
    if file is None:
        return "⚠️  Please upload a PDF file first."
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            with open(file.name, "rb") as f:
                tmp.write(f.read())
            tmp_path = tmp.name

        current_chain["chain"] = build_qa_chain(tmp_path)
        os.unlink(tmp_path)
        return "✅  PDF uploaded and indexed! Ask me anything about it."
    except Exception as exc:
        return f"❌  Error processing PDF: {exc}"


def answer_question(message: str, history: list) -> tuple:
    """Gradio callback: answer a user query using the RAG chain."""
    if not message.strip():
        return history, ""
    if current_chain["chain"] is None:
        return history + [(message, "⚠️  Please upload and index a PDF first.")], ""
    try:
        result  = current_chain["chain"]({"query": message})
        answer  = result["result"].strip()
        sources = result.get("source_documents", [])
        if sources:
            pages_used = sorted({doc.metadata.get("page", "?") for doc in sources})
            answer += f"\n\n📄 Source pages: {pages_used}"
    except Exception as exc:
        answer = f"❌  Error: {exc}"
    return history + [(message, answer)], ""


# ─── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="AI RAG Assistant – Quest Analytics",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.Markdown(
        """
        # 🔬 Task 6 – QA Bot Interface | Quest Analytics RAG Assistant
        ### Powered by LangChain + local HuggingFace models (`google/flan-t5-base`)
        > **Assignment:** Upload the research PDF, then ask: *"What this paper is talking about?"*

        **Steps:** ① Upload PDF → ② Click Index PDF → ③ Type question → ④ Click Send
        """
    )

    with gr.Row():
        # ── Left panel – upload ───────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Step 1 – Upload Your PDF")
            pdf_input     = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_btn    = gr.Button("📤 Index PDF", variant="primary")
            upload_status = gr.Textbox(
                label="Indexing Status",
                value="⬆️  Upload a PDF to begin.",
                interactive=False,
                lines=3,
            )

        # ── Right panel – chat ────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Step 2 – Ask Questions About the Document")
            chatbot  = gr.Chatbot(
                label="RAG Assistant – Document Q&A",
                height=420,
            )
            with gr.Row():
                question_box = gr.Textbox(
                    placeholder='Type your question, e.g. "What this paper is talking about?"',
                    label="Your Question",
                    lines=2,
                    scale=4,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)
            clear_btn = gr.Button("🗑 Clear chat", size="sm")

    # ── Event wiring ─────────────────────────────────────────────────────
    upload_btn.click(fn=upload_pdf,         inputs=pdf_input,
                     outputs=upload_status, api_name=False)
    send_btn.click(  fn=answer_question,    inputs=[question_box, chatbot],
                     outputs=[chatbot, question_box], api_name=False)
    question_box.submit(fn=answer_question, inputs=[question_box, chatbot],
                        outputs=[chatbot, question_box], api_name=False)
    clear_btn.click( fn=lambda: ([], ""),   outputs=[chatbot, question_box],
                     api_name=False)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # accessible via Codespace port forwarding
        server_port=7860,
        share=False,             # avoid flaky gradio.live tunnel (404/504)
    )

