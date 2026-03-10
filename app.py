"""
app.py – Standalone Gradio QA Bot
Quest Analytics – AI RAG Assistant  (Free – HuggingFace)

Run:
    python app.py

Then open the printed URL in your browser.
Requires a FREE HuggingFace token in .env:
  HUGGINGFACEHUB_API_TOKEN=hf_...
Get one free at: https://huggingface.co/settings/tokens
"""

import os, tempfile
from dotenv import load_dotenv

load_dotenv()

# ─── Credentials ──────────────────────────────────────────────────────────────
# Free HuggingFace token – get yours at https://huggingface.co/settings/tokens
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# ─── Imports ──────────────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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

# ─── LLM (free HuggingFace Hub) ───────────────────────────────────────────────
llm_parameters = {
    "max_new_tokens" : 512,
    "temperature"    : 0.7,
    "top_p"          : 0.95,
    "do_sample"      : True,
}

llm = HuggingFaceHub(
    repo_id                  = "mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token = HF_TOKEN,
    model_kwargs             = llm_parameters,
)

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
        history = history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": "⚠️  Please upload and index a PDF first."}]
        return history, ""
    try:
        result  = current_chain["chain"]({"query": message})
        answer  = result["result"].strip()
        sources = result.get("source_documents", [])
        if sources:
            pages_used = sorted({doc.metadata.get("page", "?") for doc in sources})
            answer += f"\n\n📄 Source pages: {pages_used}"
    except Exception as exc:
        answer = f"❌  Error: {exc}"
    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": answer}]
    return history, ""


# ─── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="AI RAG Assistant – Quest Analytics",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.Markdown(
        """
        # 🔬 AI RAG Assistant – Quest Analytics
        **Powered by LangChain + HuggingFace (mistralai/Mistral-7B-Instruct-v0.2)**

        **How to use:**
        1. Upload a research paper PDF on the left and click **📤 Index PDF**
        2. Type your question and click **Send** (or press Enter)
        """
    )

    with gr.Row():
        # ── Left panel – upload ───────────────────────────────────────────
        with gr.Column(scale=1):
            pdf_input     = gr.File(label="📁 Upload PDF", file_types=[".pdf"])
            upload_btn    = gr.Button("📤 Index PDF", variant="primary")
            upload_status = gr.Textbox(
                label="Status",
                value="Upload a PDF to begin.",
                interactive=False,
                lines=3,
            )

        # ── Right panel – chat ────────────────────────────────────────────
        with gr.Column(scale=2):
            chatbot  = gr.Chatbot(
                label="Chat with your Document",
                height=420,
                type="messages",
            )
            with gr.Row():
                question_box = gr.Textbox(
                    placeholder='e.g. "What this paper is talking about?"',
                    label="Your Question",
                    lines=2,
                    scale=4,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)
            clear_btn = gr.Button("🗑 Clear chat", size="sm")

    # ── Event wiring ─────────────────────────────────────────────────────
    upload_btn.click(fn=upload_pdf,       inputs=pdf_input,
                     outputs=upload_status)
    send_btn.click(  fn=answer_question,  inputs=[question_box, chatbot],
                     outputs=[chatbot, question_box])
    question_box.submit(fn=answer_question, inputs=[question_box, chatbot],
                        outputs=[chatbot, question_box])
    clear_btn.click( fn=lambda: ([], ""), outputs=[chatbot, question_box])


if __name__ == "__main__":
    demo.launch(share=True)  # share=True prints a public ngrok URL

