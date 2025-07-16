# main.py
import streamlit as st
import os
import subprocess
import json
import time
import re
from dotenv import load_dotenv
import google.generativeai as genai

# Import our custom modules
from transcribe import run_transcription
from rag_processor import extract_text_from_pdfs, build_vector_store, get_conversational_rag_answer
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma


# --- Constants for Persistence ---
PERSISTENT_DATA_DIR = "cortex_data"
SESSIONS_DIR = os.path.join(PERSISTENT_DATA_DIR, "sessions")
VECTOR_STORE_DIR = os.path.join(PERSISTENT_DATA_DIR, "vector_store")

os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)


# --- Page Configuration ---
st.set_page_config(
    page_title="Cortex AI: The intelligent memory for your team.",
    page_icon="🧠",
    layout="wide"
)

# --- Load API Keys ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()
genai.configure(api_key=google_api_key)


# --- Helper Functions ---

def sanitize_filename(name):
    """Sanitizes a filename to be used as a directory name."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

def get_available_sessions():
    """
    Scans the sessions directory and returns a dictionary mapping
    sanitized directory names to their metadata (display_name, has_audio, has_docs).
    """
    sessions = {}
    if not os.path.exists(SESSIONS_DIR):
        return sessions
    for session_dir in os.listdir(SESSIONS_DIR):
        if os.path.isdir(os.path.join(SESSIONS_DIR, session_dir)):
            meta_path = os.path.join(SESSIONS_DIR, session_dir, "session_meta.json")
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    sessions[session_dir] = {
                        "display_name": meta.get("display_name", session_dir),
                        "has_audio": meta.get("has_audio", False),
                        "has_docs": meta.get("has_docs", False)
                    }
            except (FileNotFoundError, json.JSONDecodeError):
                sessions[session_dir] = {"display_name": session_dir, "has_audio": True, "has_docs": False} # Fallback
    return sessions

def load_session_data(session_name):
    """Loads all data for a given session into the session_state."""
    st.session_state.clear()

    session_path = os.path.join(SESSIONS_DIR, session_name)
    
    meta_path = os.path.join(session_path, "session_meta.json")
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            st.session_state.display_name = meta.get("display_name", session_name)
    except (FileNotFoundError, json.JSONDecodeError):
        st.session_state.display_name = session_name
    
    meeting_dashboard_path = os.path.join(session_path, "meeting_dashboard.json")
    if os.path.exists(meeting_dashboard_path):
        with open(meeting_dashboard_path, 'r') as f:
            st.session_state.meeting_dashboard = json.load(f)

    doc_dashboard_path = os.path.join(session_path, "document_dashboard.json")
    if os.path.exists(doc_dashboard_path):
        with open(doc_dashboard_path, 'r') as f:
            st.session_state.document_dashboard = json.load(f)
            
    transcript_path = os.path.join(session_path, "transcript.txt")
    if os.path.exists(transcript_path):
        with open(transcript_path, 'r') as f:
            st.session_state.transcript = f.read()

    session_vector_store_path = os.path.join(VECTOR_STORE_DIR, session_name)
    if os.path.exists(session_vector_store_path):
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vector_store = Chroma(
            persist_directory=session_vector_store_path,
            embedding_function=embedding_function
        )
    
    st.session_state.session_loaded = True
    st.session_state.selected_session = session_name

# Other helper functions like generate_dashboard remain the same...
def generate_dashboard(transcript_text, session_name):
    """Generates the meeting dashboard and saves it to the session directory."""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt_text = f"""
    You are a professional secretary analyzing a meeting transcript. Based on the following transcript, please generate a JSON object with three keys: "summary", "topics", and "actions".
    1.  **summary**: Write a concise, professional executive summary of the key discussion points and final outcomes.
    2.  **topics**: Create a string with a bulleted list of the key topics discussed. For each topic, briefly state the main outcome or decision.
    3.  **actions**: Create a string with a Markdown table of all explicit action items. The table should have columns for 'Action Item', 'Assigned To', and 'Due Date'. If a value is not mentioned, use 'Not specified'.
    Here is the transcript: --- {transcript_text} ---
    Provide ONLY the raw JSON object as your response, without any surrounding text or markdown formatting.
    """
    generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
    try:
        response = model.generate_content(prompt_text, generation_config=generation_config)
        dashboard_content = json.loads(response.text)
        session_path = os.path.join(SESSIONS_DIR, session_name)
        with open(os.path.join(session_path, "meeting_dashboard.json"), 'w') as f:
            json.dump(dashboard_content, f, indent=4)
        return dashboard_content
    except Exception as e:
        st.error(f"Error generating AI dashboard: {e}")
        return {"summary": "Error", "topics": "Error", "actions": "Error"}


def generate_document_dashboard(all_pdf_text, session_name):
    """Generates a summary for the provided PDF documents and saves it."""
    if not all_pdf_text.strip(): return None
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt_text = f"""
    You are an AI research assistant. Analyze the combined text from the following documents. Generate a JSON object with two keys: "overall_summary" and "key_themes".
    1.  **overall_summary**: Write a concise, high-level summary of the main purpose and content of the provided documents.
    2.  **key_themes**: Create a string with a bulleted list of the most important topics, concepts, or data points found across the documents.
    Here is the combined text from the documents: --- {all_pdf_text} ---
    Provide ONLY the raw JSON object as your response, without any surrounding text or markdown formatting.
    """
    generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
    try:
        response = model.generate_content(prompt_text, generation_config=generation_config)
        dashboard_content = json.loads(response.text)
        session_path = os.path.join(SESSIONS_DIR, session_name)
        with open(os.path.join(session_path, "document_dashboard.json"), 'w') as f:
            json.dump(dashboard_content, f, indent=4)
        return dashboard_content
    except Exception as e:
        st.error(f"Error generating document dashboard: {e}")
        return {"overall_summary": "Error", "key_themes": "Error"}


# --- UI Layout ---
st.title("Cortex AI 🧠")
st.header("The intelligent memory for your team.")
st.markdown("Upload meeting files to create a new session, or load a previous session to continue your work.")

# --- Sidebar for Session Management and File Uploads ---
with st.sidebar:
    st.header("1. Choose a Session")
    
    sessions_dict = get_available_sessions()
    
    def on_session_change():
        sanitized_name = st.session_state.session_selector
        if sanitized_name:
            load_session_data(sanitized_name)
        else: st.session_state.clear()

    # CHANGED: format_func now adds icons based on session content
    def format_session_name(key):
        if not key:
            return "--- Select a Session ---"
        meta = sessions_dict.get(key, {})
        display_name = meta.get("display_name", key)
        icons = []
        if meta.get("has_audio"):
            icons.append("🎙️")
        if meta.get("has_docs"):
            icons.append("📄")
        return f"{' '.join(icons)} {display_name}"

    st.selectbox(
        "Load Existing Session",
        options=[""] + list(sessions_dict.keys()),
        format_func=format_session_name,
        key="session_selector",
        on_change=on_session_change,
        help="Select a previously processed session to load its data."
    )
    
    st.divider()

    st.header("2. Process New Files")
    # CHANGED: Uploader now accepts video formats
    media_file = st.file_uploader(
        "Upload Audio or Video File",
        type=["wav", "mp3", "m4a", "mp4", "mov", "mkv", "avi", "webm"]
    )
    pdf_files = st.file_uploader("Upload PDFs (optional)", type="pdf", accept_multiple_files=True)

    process_button = st.button("🚀 Process Files", disabled=(not media_file and not pdf_files))

# --- Main Processing Logic ---
if process_button:
    if media_file:
        display_name = os.path.splitext(media_file.name)[0]
    else: 
        display_name = os.path.splitext(pdf_files[0].name)[0]
    
    sanitized_name = sanitize_filename(display_name)
    
    if sanitized_name in get_available_sessions():
        st.warning(f"Session '{display_name}' already exists. Loading it instead.")
        load_session_data(sanitized_name)
        st.rerun()

    st.session_state.clear() 
    st.session_state.selected_session = sanitized_name
    st.session_state.display_name = display_name

    with st.spinner(f"Processing new session: '{display_name}'... This may take a few minutes."):
        temp_dir = f"temp_{sanitized_name}"
        os.makedirs(temp_dir, exist_ok=True)
        session_path = os.path.join(SESSIONS_DIR, sanitized_name)
        os.makedirs(session_path, exist_ok=True)

        # CHANGED: Save enhanced metadata about session contents
        with open(os.path.join(session_path, "session_meta.json"), 'w') as f:
            meta_data = {
                "display_name": display_name,
                "has_audio": media_file is not None,
                "has_docs": bool(pdf_files)
            }
            json.dump(meta_data, f)
        
        transcript_text = ""
        if media_file:
            original_media_path = os.path.join(temp_dir, media_file.name)
            with open(original_media_path, "wb") as f: f.write(media_file.getbuffer())
            wav_filepath = os.path.join(temp_dir, f"{os.path.splitext(media_file.name)[0]}.wav")
            try:
                # CHANGED: Added '-vn' to robustly extract audio from video files
                subprocess.run(
                    ['ffmpeg', '-i', original_media_path, '-vn', '-ar', '16000', '-ac', '1', '-y', wav_filepath],
                    check=True, capture_output=True, text=True
                )
            except subprocess.CalledProcessError as e:
                st.error(f"FFmpeg error: {e.stderr}"); st.stop()
            transcript_text = run_transcription(wav_filepath)
            st.session_state.transcript = transcript_text
            with open(os.path.join(session_path, "transcript.txt"), 'w') as f: f.write(transcript_text)

        all_pdf_text, documents_data = "", []
        if pdf_files:
            pdf_paths = []
            for pdf in pdf_files:
                pdf_path = os.path.join(temp_dir, pdf.name)
                with open(pdf_path, "wb") as f: f.write(pdf.getbuffer())
                pdf_paths.append(pdf_path)
            documents_data = extract_text_from_pdfs(pdf_paths)
            all_pdf_text = "\n\n".join([doc.page_content for doc in documents_data])

        session_vector_store_path = os.path.join(VECTOR_STORE_DIR, sanitized_name)
        st.session_state.vector_store = build_vector_store(
            pdf_docs=documents_data, transcript_text=transcript_text, persist_directory=session_vector_store_path)

        if transcript_text:
            st.session_state.meeting_dashboard = generate_dashboard(transcript_text, sanitized_name)
        if all_pdf_text:
            st.session_state.document_dashboard = generate_document_dashboard(all_pdf_text, sanitized_name)

        st.session_state.session_loaded = True
        st.success(f"✅ Processing complete! Session '{display_name}' is saved and ready.")
        time.sleep(2)
        st.rerun()

# --- Main Content Area for Display and Chat ---
if st.session_state.get("session_loaded"):
    st.header(f"Session: {st.session_state.get('display_name', 'Untitled Session')}")
    # ... The rest of the display logic is unchanged ...
    meeting_dashboard = st.session_state.get("meeting_dashboard")
    doc_dashboard = st.session_state.get("document_dashboard")
    
    if meeting_dashboard:
        st.subheader("Meeting Dashboard")
        summary, topics, actions, transcript_tab = st.tabs(["📄 Summary", "📌 Key Topics", "✅ Action Items", "📝 Full Transcript"])
        with summary: st.markdown(meeting_dashboard.get("summary", "Not available."))
        with topics: st.markdown(meeting_dashboard.get("topics", "Not available."))
        with actions: st.markdown(meeting_dashboard.get("actions", "Not available."))
        with transcript_tab:
            with st.expander("Click to view the full meeting transcript"):
                 st.text_area("", value=st.session_state.get('transcript', 'Transcript not available.'), height=400, disabled=True)

    if doc_dashboard:
        st.subheader("Document Dashboard")
        doc_summary, doc_themes = st.tabs(["📑 Overall Summary", "🔑 Key Themes"])
        with doc_summary: st.markdown(doc_dashboard.get("overall_summary", "Not available."))
        with doc_themes: st.markdown(doc_dashboard.get("key_themes", "Not available."))
        
    st.divider()

    st.subheader("💬 Chat with your Assistant")
    if not st.session_state.get("vector_store"):
        st.warning("Cannot start chat without a loaded knowledge base.")
    else:
        if "chat_session" not in st.session_state:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            st.session_state.chat_session = model.start_chat(history=[])
        for message in st.session_state.chat_session.history:
            role = "assistant" if message.role == "model" else message.role
            with st.chat_message(role): st.markdown(message.parts[0].text)
        if prompt := st.chat_input("Ask about the content of the indexed files..."):
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_conversational_rag_answer(
                        prompt, st.session_state.vector_store, st.session_state.chat_session)
                    st.markdown(response)
else:
    st.info("Please upload files to start a new session or select a saved session from the sidebar.")

