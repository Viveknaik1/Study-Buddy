#!/usr/bin/env python3
"""
Study Buddy - Enhanced Version with Smart Persistence

Enhanced with:
- Automatic index loading on startup (no button needed!)
- Chat history persistence across sessions
- File state memory (remembers uploaded files)
- Faster startup and improved performance
- Clean ChatGPT-like interface
- Smart caching for instant responses
- Single optimized embedding model (all-MiniLM-L6-v2)

Run instructions:
1. Copy env.example to .env and set GOOGLE_API_KEY (optional for better chat)
2. pip install -r requirements.txt  
3. Run app: streamlit run main_enhanced.py
"""

import os
import time
import asyncio
import json
from typing import List, Optional
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

# Other imports
import PyPDF2
from sentence_transformers import SentenceTransformer
import warnings
import re

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Persistence files
CHAT_HISTORY_FILE = "chat_history.json"
APP_STATE_FILE = "app_state.json"


class LocalEmbeddings(Embeddings):
    """Local CPU embeddings using all-MiniLM-L6-v2 for optimal performance."""
    
    def __init__(self):
        try:
            self.model_name = EMBEDDING_MODEL
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            self.model.max_seq_length = 512
        except Exception as e:
            st.error(f"Error loading embedding model: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with local model."""
        try:
            if not texts:
                return []
            clean_texts = [self._clean_text(text) for text in texts if text.strip()]
            if not clean_texts:
                return []
            
            embeddings = self.model.encode(
                clean_texts,
                batch_size=16,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        except Exception as e:
            st.error(f"Error embedding documents: {e}")
            return []
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with local model."""
        try:
            if not text.strip():
                return []
            clean_text = self._clean_text(text)
            embedding = self.model.encode(
                [clean_text],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding[0].tolist()
        except Exception as e:
            st.error(f"Error embedding query: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better embedding."""
        text = re.sub(r'\s+', ' ', text.strip())
        return text[:512] if len(text) > 512 else text


# Single optimal embedding model for document Q&A
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def save_chat_history(messages):
    """Save chat history to file."""
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_chat_history():
    """Load chat history from file."""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_app_state(uploaded_files_list):
    """Save app state to file."""
    try:
        state = {
            "uploaded_files_list": uploaded_files_list,
            "timestamp": time.time()
        }
        with open(APP_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_app_state():
    """Load app state from file."""
    try:
        if os.path.exists(APP_STATE_FILE):
            with open(APP_STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
                # Check if state is recent (within 7 days)
                if time.time() - state.get("timestamp", 0) < 604800:
                    return state
    except Exception:
        pass
    return {}


@st.cache_resource(show_spinner=False)
def get_local_embeddings():
    """Get cached local embeddings model."""
    return LocalEmbeddings(EMBEDDING_MODEL)


def ensure_event_loop():
    """Ensure asyncio event loop exists for Windows compatibility."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


@st.cache_resource(show_spinner=False)
def get_chat_model(api_key: str):
    """Get cached Google Gemini chat model."""
    ensure_event_loop()
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        google_api_key=api_key,
        transport="rest",
        request_options={"timeout": 60}
    )


def extract_text_from_pdf_silent(file_content: bytes, filename: str) -> str:
    """Extract text from PDF silently without progress messages."""
    try:
        import io
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_parts = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if not page_text.strip():
                    try:
                        page_text = page.extract_text(extraction_mode="layout")
                    except:
                        page_text = page.extractText() if hasattr(page, 'extractText') else ""
                
                if page_text.strip():
                    cleaned_text = re.sub(r'\s+', ' ', page_text.strip())
                    text_parts.append(f"--- Page {page_num + 1} ---\n{cleaned_text}\n")
                    
            except Exception:
                continue
        
        return "\n".join(text_parts)
            
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_content: bytes, filename: str) -> str:
    """Extract text from PDF file content."""
    return extract_text_from_pdf_silent(file_content, filename)


@st.cache_data(show_spinner=False)
def extract_text_from_txt(file_content: bytes, filename: str) -> str:
    """Extract text from TXT file content."""
    try:
        text = file_content.decode('utf-8')
        return text
    except UnicodeDecodeError:
        encodings = ['latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                text = file_content.decode(encoding)
                return text
            except:
                continue
        return ""
    except Exception:
        return ""


def process_uploaded_files_silent(uploaded_files) -> List[Document]:
    """Process uploaded files silently without detailed messages."""
    documents = []
    
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        file_content = uploaded_file.read()
        
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_content, filename)
        elif filename.endswith('.txt'):
            text = extract_text_from_txt(file_content, filename)
        else:
            continue
        
        if text.strip():
            clean_text = re.sub(r'\s+', ' ', text.strip())
            if len(clean_text) > 50:
                doc = Document(
                    page_content=clean_text,
                    metadata={"source": filename, "length": len(clean_text)}
                )
                documents.append(doc)
    
    return documents


@st.cache_data(show_spinner=False)
def split_documents(_documents: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n--- Page", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return text_splitter.split_documents(_documents)


def build_vectorstore_silent(texts: List[Document], embeddings) -> Optional[FAISS]:
    """Build FAISS vectorstore silently with simple progress."""
    try:
        if not texts:
            return None
            
        progress_bar = st.progress(0)
        batch_size = 20
        vectorstore = None
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            progress = (i + len(batch)) / len(texts)
            progress_bar.progress(progress)
            
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    batch_store = FAISS.from_documents(batch, embeddings)
                    vectorstore.merge_from(batch_store)
            except Exception:
                continue
        
        progress_bar.progress(1.0)
        time.sleep(0.5)
        progress_bar.empty()
        
        return vectorstore
    except Exception:
        return None


def auto_load_index():
    """Automatically load saved index on startup."""
    index_dir = "faiss_index"
    if not os.path.exists(index_dir):
        return None
    
    try:
        embeddings = get_local_embeddings()
        vectorstore = FAISS.load_local(
            index_dir, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception:
        return None


def retrieve_documents(vectorstore, query: str, k: int = 5):
    """Retrieve documents using optimized search."""
    try:
        docs = vectorstore.max_marginal_relevance_search(
            query, 
            k=k, 
            fetch_k=k*3, 
            lambda_mult=0.7
        )
        return docs
    except Exception:
        try:
            docs = vectorstore.similarity_search(query, k=k)
            return docs
        except Exception:
            return []


def get_ai_response(question: str) -> dict:
    """Generate AI response using local embeddings + Google chat."""
    try:
        docs = retrieve_documents(st.session_state.vectorstore, question)
        
        if not docs:
            return {
                "answer": "I couldn't find relevant information in your documents for this question. Try rephrasing your question or check if the information exists in your uploaded documents.",
                "sources": ""
            }
        
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if api_key:
            try:
                chat_model = get_chat_model(api_key)
                
                template = """You are a helpful assistant that answers questions based solely on the provided document context.

INSTRUCTIONS:
1. Answer the question using ONLY the information from the provided context
2. If the context doesn't contain enough information, say "I don't have enough information in the uploaded documents to answer this question."
3. Be specific and cite relevant details from the context
4. Keep your answer clear and concise
5. Don't make up information not present in the context
6. Be conversational and friendly in your response

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
                
                qa_prompt = PromptTemplate(
                    template=template,
                    input_variables=["context", "question"]
                )
                
                chain = load_qa_chain(chat_model, chain_type="stuff", prompt=qa_prompt)
                response = chain.invoke({"input_documents": docs, "question": question})
                answer = response["output_text"]
                
            except Exception:
                answer = "Here are the relevant excerpts from your documents:\n\n"
                for i, doc in enumerate(docs[:2], 1):
                    answer += f"**Excerpt {i}:** {doc.page_content[:300]}...\n\n"
        else:
            answer = "Here are the relevant excerpts from your documents:\n\n"
            for i, doc in enumerate(docs[:2], 1):
                answer += f"**Excerpt {i}:** {doc.page_content[:300]}...\n\n"
        
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
        source_text = ", ".join(sources)
        
        return {
            "answer": answer,
            "sources": source_text
        }
        
    except Exception:
        return {
            "answer": "I encountered an error while processing your question.",
            "sources": ""
        }


def main():
    """Main Streamlit application with enhanced persistence."""
    load_dotenv()
    
    # Page configuration
    st.set_page_config(
        page_title="Study Buddy",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for clean design
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stChatMessage {
        background-color: transparent;
    }
    .stChatInput {
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state with smart loading
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        
        # Load previous app state
        previous_state = load_app_state()
        
        # Initialize with previous state or defaults
        st.session_state.uploaded_files_list = previous_state.get("uploaded_files_list", [])
        st.session_state.messages = load_chat_history()
        
        # Try to auto-load saved index
        vectorstore = auto_load_index()
        st.session_state.vectorstore = vectorstore
        
        # Show welcome message if restored
        if vectorstore and st.session_state.uploaded_files_list:
            st.sidebar.success(f"⚡ Restored {len(st.session_state.uploaded_files_list)} files")
        elif vectorstore:
            st.sidebar.success("⚡ Index loaded automatically")
    
    # ===== SIDEBAR =====
    with st.sidebar:
        st.title("Study Buddy")
        st.markdown("*Enhanced with Smart Persistence*")
        
        st.markdown("---")
        
        # Model Info
        st.subheader("AI Model")
        st.info(f"Embeddings: {EMBEDDING_MODEL}")
        st.caption("Optimized for fast document Q&A")
        
        st.markdown("---")
        
        # File Upload
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        # Document List
        if st.session_state.uploaded_files_list:
            st.subheader("Documents")
            for i, filename in enumerate(st.session_state.uploaded_files_list, 1):
                st.text(f"{i}. {filename}")
        
        st.markdown("---")
        
        # Action Buttons - Simplified since auto-loading works
        if st.button("Clear All", use_container_width=True):
            st.session_state.uploaded_files_list = []
            st.session_state.vectorstore = None
            st.session_state.messages = []
            # Clear saved files
            for file in [CHAT_HISTORY_FILE, APP_STATE_FILE]:
                if os.path.exists(file):
                    os.remove(file)
            # Clear faiss index
            import shutil
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
            st.rerun()
        
        st.markdown("---")
        
        # Status
        if st.session_state.uploaded_files_list:
            st.success(f"{len(st.session_state.uploaded_files_list)} files ready")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            st.success("AI chat enabled")
        else:
            st.info("Add GOOGLE_API_KEY for AI responses")
        
        # Performance indicator
        if st.session_state.vectorstore:
            st.success("⚡ Fast mode active")
        
        # Show persistence status
        if os.path.exists(CHAT_HISTORY_FILE):
            st.info("💾 Chat history saved")
    
    # ===== MAIN AREA =====
    st.header("Study Buddy")
    st.markdown("Ask questions about your documents")
    
    # Process uploaded files
    if uploaded_files:
        current_files = [f.name for f in uploaded_files]
        if current_files != st.session_state.uploaded_files_list:
            st.session_state.uploaded_files_list = current_files
            
            with st.spinner("Processing documents..."):
                documents = process_uploaded_files_silent(uploaded_files)
                
                if documents:
                    texts = split_documents(documents)
                    embeddings = get_local_embeddings()
                    vectorstore = build_vectorstore_silent(texts, embeddings)
                    
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        # Save everything automatically
                        try:
                            vectorstore.save_local("faiss_index")
                            with open("faiss_index/backend.txt", "w") as f:
                                f.write(EMBEDDING_MODEL)
                            save_app_state(st.session_state.uploaded_files_list)
                        except Exception:
                            pass
                        st.success(f"Successfully processed {len(current_files)} files!")
                        time.sleep(1)
                        st.rerun()
    
    # Chat interface
    if not st.session_state.vectorstore:
        st.info("Upload documents to get started")
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "sources" in message and message["sources"]:
                    st.caption(f"Sources: {message['sources']}")
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_ai_response(prompt)
                    st.write(response["answer"])
                    if response["sources"]:
                        st.caption(f"Sources: {response['sources']}")
                
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["answer"],
                    "sources": response["sources"]
                })
            
            # Auto-save chat history after each interaction
            save_chat_history(st.session_state.messages)


if __name__ == "__main__":
    main()
