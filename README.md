# Study Buddy - Enhanced RAG Chat Application

A powerful Retrieval-Augmented Generation (RAG) application built with Streamlit that transforms how you interact with your documents. Upload PDFs and text files, get instant AI-powered answers with source citations, and enjoy persistent sessions that remember everything.

![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2.1-green)
![FAISS](https://img.shields.io/badge/FAISS-1.8.0-orange)

## Key Features

- ** Smart Persistence**: Automatically restores your chat history, uploaded files, and pre-built FAISS index across sessions
- ** Fast Local Embeddings**: Uses `all-MiniLM-L6-v2` for optimal speed and accuracy in document Q&A
- ** Multi-Format Support**: Upload PDF and TXT files with intelligent text extraction
- ** AI-Powered Responses**: Optional Google Gemini integration for enhanced answer quality
- ** Intelligent Search**: MMR-based retrieval with FAISS for diverse, relevant results
- ** Session Memory**: Never lose your work - everything is automatically saved and restored

## 🛠️ Tech Stack

- **Frontend**: Streamlit (1.35.0)
- **AI/ML**: LangChain, FAISS, Sentence Transformers
- **Language Models**: Google Gemini (optional), Local Embeddings
- **Document Processing**: PyPDF2, Text Processing
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Environment**: Python 3.8+, Virtual Environment

##  Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Study-Buddy-RAG.git
   cd Study-Buddy-RAG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example environment file
   copy env.example .env
   
   # Edit .env and add your Google API key (optional)
   GOOGLE_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-1.5-flash
   ```

5. **Run the application**
   ```bash
   streamlit run main_enhanced.py
   ```

6. **Open your browser** and navigate to `http://localhost:8501`

## 📖 How to Use

### Basic Usage
1. **Upload Documents**: Use the sidebar to upload PDF or TXT files
2. **Wait for Processing**: The app automatically processes and indexes your documents
3. **Ask Questions**: Type questions in the chat interface
4. **Get Answers**: Receive AI-generated responses with source citations

### Advanced Features
- **Bulk Ingestion**: Use `python ingest.py` to process multiple files from the `data/` directory
- **Model Selection**: The app automatically uses the optimal `all-MiniLM-L6-v2` embedding model
- **Persistence**: Your chat history and uploaded files are automatically saved and restored

##  Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google API key for Gemini integration (optional)
- `GEMINI_MODEL`: Gemini model to use (default: `gemini-1.5-flash`)

### Embedding Model
The application uses `all-MiniLM-L6-v2` by default, which provides:
- Fast inference (384-dimensional embeddings)
- Excellent semantic similarity for documents
- Balanced speed vs. quality trade-off

## 📁 Project Structure

```
Study-Buddy-RAG/
├── main_enhanced.py          # Main Streamlit application
├── ingest.py                 # CLI tool for bulk document ingestion
├── requirements.txt          # Python dependencies
├── env.example              # Environment variables template
├── .gitignore               # Git ignore file
├── README.md                # This file
├── data/                    # Directory for bulk document ingestion
├── faiss_index/             # FAISS vector database (auto-generated)
├── chat_history.json        # Chat session persistence (auto-generated)
└── app_state.json           # Application state persistence (auto-generated)
```

##  Use Cases

- **Academic Research**: Quickly search through research papers and academic documents
- **Business Intelligence**: Analyze reports, contracts, and business documents
- **Legal Research**: Search through legal documents and case files
- **Content Creation**: Research and reference materials for writing projects
- **Personal Knowledge Management**: Organize and search through personal documents

##  How It Works

1. **Document Ingestion**: PDFs and TXTs are processed and text is extracted
2. **Text Chunking**: Documents are split into manageable chunks with overlap
3. **Embedding Generation**: Text chunks are converted to vector embeddings using `all-MiniLM-L6-v2`
4. **Vector Storage**: Embeddings are stored in a FAISS index for fast similarity search
5. **Query Processing**: User questions are embedded and matched against stored vectors
6. **Response Generation**: Relevant document chunks are retrieved and used to generate AI responses
7. **Persistence**: All data is automatically saved for future sessions

##  Performance Features

- **Smart Caching**: Embeddings and models are cached for faster subsequent runs
- **Batch Processing**: Documents are processed in batches for memory efficiency
- **Optimized Search**: MMR (Maximal Marginal Relevance) for diverse, relevant results
- **Session Persistence**: No need to re-upload or re-index documents
