## 🤖 Gemini-Powered RAG Chatbot for Multi-format Document Querying and Summarization

This project is an intelligent, multi-format document assistant powered by **Gemini (Google Generative AI)** and **RAG (Retrieval-Augmented Generation)** using **LangChain**, **FAISS**, and **Hugging Face embeddings**. It supports PDF, DOCX, TXT, Markdown, and image files (JPG, PNG) — providing question answering and summarization capabilities via a **Gradio web interface**.

---

## 🚀 Features

- 📄 **Multi-format support**: Upload PDF, DOCX, TXT, Markdown, JPG, or PNG.
- 🔍 **RAG-based chatbot**: Combines document retrieval and Gemini's LLM to answer user queries.
- 🧠 **Vector-based semantic search**: Uses FAISS and HuggingFace `MiniLM` embeddings.
- 📝 **Document summarization**: Automatically summarizes uploaded documents.
- 🖼️ **OCR for image documents**: Extracts and summarizes text from scanned documents or photos.
- 💬 **Conversational memory**: Maintains chat history for context-aware responses.
- 🌐 **Gradio UI**: Simple and interactive frontend to upload and chat.

---

## 📁 Project Structure

gemini-rag-chatbot/
├── app.py # Main application code (Gradio + RAG setup)
├── .env # Stores HF and Gemini API keys
├── requirements.txt # All dependencies listed here
├── faiss_index/ # Saved FAISS vector store
├── uploaded_docs/ # Folder for uploaded documents


---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/gemini-rag-chatbot.git
cd gemini-rag-chatbot

```
### 2. Create & Activate a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. API Keys
Create a .env file in the root directory and add your keys:
```bash
HF_TOKEN=your_huggingface_token
GEMINI_API_KEY=your_gemini_api_key
```


### 5. 📷 Image OCR Setup (Windows)
If you plan to use image support (JPG, PNG), install Tesseract OCR and add its path:

✅ Download:
https://github.com/tesseract-ocr/tesseract

⚙️ Configure path:
In app.py, update:
```bash
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### 6.  Run the App

```bash
python app.py
```

###


### Technologies Used
Tool	                  Purpose
LangChain           	RAG orchestration
FAISS               	Vector similarity search
Gemini 1.5 Flash	    Large language model
HuggingFace Hub      	Embedding model (MiniLM-L6-v2)
pytesseract	          OCR for image-to-text
Gradio	              Web interface for uploading and chatting
dotenv	              API key management

