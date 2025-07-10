
# --- Setup & Configuration ---
from dotenv import load_dotenv
import os, shutil, traceback

load_dotenv()
from huggingface_hub import login

# Load API keys from .env
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HF_TOKEN"]
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

login(token=os.environ["HF_TOKEN"])

# --- Gemini LLM Setup ---
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    convert_system_message_to_human=True,
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.5
)


# --- Document Loaders ---
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader,
    UnstructuredMarkdownLoader, UnstructuredImageLoader
)

def load_docs(file_path):
    ext = file_path.split('.')[-1].lower()
    loader_map = {
        "pdf": PyPDFLoader,
        "docx": Docx2txtLoader,
        "txt": TextLoader,
        "md": UnstructuredMarkdownLoader,
        "jpg": UnstructuredImageLoader,
        "jpeg": UnstructuredImageLoader,
        "png": UnstructuredImageLoader
    }
    if ext in loader_map:
        return loader_map[ext](file_path).load()
    raise ValueError(f"Unsupported file format: {ext}")

# --- Text Splitting, Embedding & Vector Store ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embed_and_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectordb = FAISS.from_documents(chunks, embedding_model)

    faiss_path = r"D:\rag_chatbot\gemini_rag_bot\faiss_index"
    vectordb.save_local(faiss_path)
    print("✅ FAISS index saved at", faiss_path)
    return vectordb

# --- RAG Chain with Gemini ---
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

rag_chain = None

def build_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

# --- Summarizer & OCR for Images ---
from langchain.chains.summarize import load_summarize_chain
from PIL import Image
import pytesseract

def summarize_docs(docs):
    try:
        summary_chain = load_summarize_chain(llm, chain_type="stuff")
        return summary_chain.run(docs)
    except Exception:
        return "⚠️ Unable to generate summary for this document."
    
import pytesseract

# Explicitly point to tesseract.exe on Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    

def ocr_image_summary(file_path):
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
        if len(text.strip()) < 20:
            return "⚠️ No significant text found in image."
        return summarize_docs([Document(page_content=text)])
    except Exception:
        return "⚠️ Unable to extract or summarize text from image."

# --- File Processor ---
def process_file(file_path):
    global rag_chain
    try:
        filename = os.path.basename(file_path)
        save_path =  rf"D:\rag_chatbot\gemini_rag_bot\uploaded_docs\{filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        shutil.copy(file_path, save_path)

        print("📄 Loading and processing document...")
        ext = filename.split('.')[-1].lower()

        if ext in ["jpg", "jpeg", "png"]:
            print("🧠 Using OCR for image document...")
            text = pytesseract.image_to_string(Image.open(save_path))
            if len(text.strip()) < 20:
                return "⚠️ No significant text found in image."
            docs = [Document(page_content=text)]
        else:
            docs = load_docs(save_path)

        print(f"📚 {len(docs)} documents loaded.")

        print("🧠 Embedding and building vector store...")
        vectordb = embed_and_store(docs)

        print("🔗 Building RAG chain...")
        rag_chain = build_chain(vectordb)

        print("📝 Summarizing document...")
        summary = summarize_docs(docs)

        return f"✅ Document processed and knowledge base built!\n\n📄 **Summary**:\n{summary}"
    
    except Exception:
        return f"❌ Error processing file:\n{traceback.format_exc()}"


# --- Chat Handler ---
def chat_with_rag(message, history):
    if rag_chain is None:
        return "⚠️ Please upload a document first."
    
    try:
        # Handle casual/generic user messages
        generic_inputs = {
            "ok": "👍 Got it!",
            "okay": "👌",
            "thanks": "You're welcome! 😊",
            "thank you": "Happy to help!",
            "cool": "😎",
            "nice": "Great!"
        }

        user_msg = message.strip().lower()
        if user_msg in generic_inputs:
            return generic_inputs[user_msg]

        # Format chat history if needed (not used here, but reserved for extensions)
        formatted_history = [(h[0], h[1]) for h in history if h[0] and h[1]]

        # Run RAG chain
        result = rag_chain({"question": message})
        answer = result.get("answer", "").strip()
        sources = result.get("source_documents", [])

        # Handle no answer or irrelevant sources
        if not answer or not sources or all(len(doc.page_content.strip()) < 20 for doc in sources):
            return "🤖 I couldn't find anything relevant in the document for your query."

        return answer

    except Exception:
        return f"❌ Error:\n{traceback.format_exc()}"
# --- Gradio UI ---
import gradio as gr
import time

time.sleep(60)  # Optional delay after setup

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Gemini-Powered RAG Chatbot\nUpload a file and ask questions!")

    with gr.Row():
        file_input = gr.File(
            label="Upload PDF, DOCX, TXT, MD, JPG, or PNG",
            type="filepath",
            file_types=[".pdf", ".docx", ".txt", ".md", ".jpg", ".jpeg", ".png"]
        )
        upload_status = gr.Textbox(label="Upload Status", interactive=False)

    file_input.change(fn=process_file, inputs=file_input, outputs=upload_status)

    chatbot = gr.ChatInterface(fn=chat_with_rag, title="💬 Ask Anything from Your Document")

demo.launch(share=True)