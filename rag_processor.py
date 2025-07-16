# rag_processor.py
import os
import pdfplumber
import nltk
import urllib.error
import shutil
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# --- NLTK Data Check ---
try:
    nltk.data.find('tokenizers/punkt')
except (urllib.error.URLError, LookupError):
    print("Downloading NLTK 'punkt' data...")
    nltk.download('punkt', quiet=True)


def extract_text_from_pdfs(pdf_paths):
    """Extracts text from a list of PDF files."""
    processed_data = []
    if not pdf_paths:
        return processed_data
        
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        full_text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                    full_text += f"\n\n--- Page {page_num} of {filename} ---\n\n" + page_text
            
            if full_text.strip():
                processed_data.append(Document(page_content=full_text.strip(), metadata={"source": filename}))
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
    return processed_data


def build_vector_store(pdf_docs, transcript_text, persist_directory):
    """
    Creates and persists a ChromaDB vector store.
    - PDF documents are split into chunks.
    - The entire transcript is treated as a single document.
    """
    print(f"\nBuilding vector store at: {persist_directory}")
    
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"  Removed existing vector store at {persist_directory}")
        
    all_chunks = []
    
    # 1. Split only the PDF documents into chunks
    if pdf_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        pdf_chunks = text_splitter.split_documents(pdf_docs)
        all_chunks.extend(pdf_chunks)
        print(f"  Split {len(pdf_docs)} PDF(s) into {len(pdf_chunks)} text chunks.")

    # 2. Add the entire transcript as a single document (if it exists)
    if transcript_text:
        transcript_doc = Document(page_content=transcript_text, metadata={"source": "meeting_transcript.txt"})
        all_chunks.append(transcript_doc)
        print("  Added the full transcript as a single chunk.")

    if not all_chunks:
        print("  No content to build vector store.")
        return None

    # 3. Create Embeddings and persist the ChromaDB vector store
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    
    print("Vector store built and persisted successfully!")
    return vector_store


def get_indexed_files(vector_store):
    """Retrieves the list of unique source filenames from the vector store."""
    if not vector_store:
        return []
    try:
        retrieved_docs = vector_store.get(include=["metadatas"])
        sources = [meta['source'] for meta in retrieved_docs['metadatas'] if 'source' in meta]
        return sorted(list(set(sources)))
    except Exception as e:
        print(f"Error getting indexed files: {e}")
        return ["Could not retrieve file list."]


def get_conversational_rag_answer(user_prompt: str, vector_store, chat_session) -> str:
    """Generates a conversational response using RAG with Gemini."""
    if not vector_store:
        return "The knowledge base is not loaded. Please process or load a session first."
    try:
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 10})
        relevant_docs = retriever.invoke(user_prompt)
        context = "\n---\n".join([doc.page_content for doc in relevant_docs])

        rag_prompt = f"""
        You are an AI meeting assistant. Your task is to answer the user's question based ONLY on the provided context below and the conversation history.
        - If the answer is in the context, provide a clear and concise answer.
        - If the answer is not in the context, state "I couldn't find information about that in the provided meeting records."
        - Do not make up information.

        CONTEXT FROM MEETING DOCUMENTS AND TRANSCRIPT:
        ---
        {context}
        ---

        USER'S QUESTION: {user_prompt}
        """

        response = chat_session.send_message(rag_prompt)
        
        return response.text

    except Exception as e:
        return f"An error occurred in the RAG chain: {e}"
