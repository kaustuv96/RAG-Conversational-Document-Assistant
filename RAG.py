#!/usr/bin/env python
# coding: utf-8

# #### Objectives
# 1. Develop an AI-Powered Document Q&A Agent with Conversational Memory using RAG.
# 2. Allow users to upload a PDF, ask questions, and receive answers grounded in the document.
# 3. Maintain conversation history for follow-up questions.
# 4. Use Gemini, LangChain, FAISS, and Streamlit.
# 5. Show retrieved context used for grounding the answer.
# 6. Handle API Key loading securely and track token usage (estimate per turn).

# #### Importing Libraries
import os
import streamlit as st
import PyPDF2
from io import BytesIO
import time # Not strictly used in current flow but good for potential timing
import traceback # For detailed error logging
import logging # For more detailed logging

# LangChain specific imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# Tokenizer for Estimation
from transformers import AutoTokenizer, logging as hf_transformers_logging
hf_transformers_logging.set_verbosity_error()

# For retries
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type, before_sleep_log
from google.api_core import exceptions as google_api_exceptions

# Setup basic logging to see in Streamlit Cloud logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Conversational Document Q&A (RAG)", layout="wide")

# --- Configuration ---
GEMINI_MODEL_NAME = "gemini-2.0-flash" # As requested by user
EMBEDDING_MODEL_NAME = "models/embedding-004"

GOOGLE_API_KEY = None
API_KEY_LOADED_FROM = None

# --- API Key Handling ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    API_KEY_LOADED_FROM = "Streamlit secrets"
except (KeyError, FileNotFoundError):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        API_KEY_LOADED_FROM = "environment variable"

if not GOOGLE_API_KEY:
    st.error("🔴 Google API Key is missing!")
    st.info("""
    Please set the GOOGLE_API_KEY:
    1.  **For local development:** Set it as an environment variable.
        You can create a `.env` file in your project root with `GOOGLE_API_KEY='YOUR_API_KEY'`
        (and install `python-dotenv`, then add `from dotenv import load_dotenv; load_dotenv()` at the script's start).
    2.  **For Streamlit Community Cloud deployment:** Add it to your app's Secrets in the settings.
    """)
    st.stop()

try:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
except ImportError:
    st.warning("`google-generativeai` library not found. This is okay if only using LangChain integrations.")
except Exception as e:
    st.warning(f"Could not configure `google.generativeai` directly: {e}")

# --- Initialize Tokenizer ---
try:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", legacy=False)
except Exception as e:
    st.warning(f"⚠️ Failed to load tokenizer: {e}. Token counts will be estimated based on word count.")
    tokenizer = None

# --- Helper Functions ---
def count_tokens(text: str) -> int:
    if not text: return 0
    if tokenizer:
        try: return len(tokenizer.encode(text))
        except Exception: return len(text.split())
    else: return len(text.split())

def extract_text_from_pdf(pdf_file_bytes):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file_bytes)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text: text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

# --- Streamlit Session State Initialization ---
if 'vector_store' not in st.session_state: st.session_state.vector_store = None
if 'conversation_chain' not in st.session_state: st.session_state.conversation_chain = None
if 'chat_history_display' not in st.session_state: st.session_state.chat_history_display = []
if 'processing_done' not in st.session_state: st.session_state.processing_done = False
if 'cumulative_input_tokens' not in st.session_state: st.session_state.cumulative_input_tokens = 0
if 'cumulative_output_tokens' not in st.session_state: st.session_state.cumulative_output_tokens = 0

def update_token_display():
    st.sidebar.markdown("### Token Usage (Cumulative Estimate)")
    st.sidebar.markdown(f"""
    **Input Tokens (Questions):** {st.session_state.cumulative_input_tokens:,}
    **Output Tokens (Answers):** {st.session_state.cumulative_output_tokens:,}
    **Total Tokens (Q+A):** {st.session_state.cumulative_input_tokens + st.session_state.cumulative_output_tokens:,}
    """)
    st.sidebar.caption("Note: Input tokens estimate user questions only, not the full context sent to the LLM.")

# ### Streamlit App Layout ---
st.title("💬 Conversational AI Document Q&A with RAG")
st.markdown(f"Upload a PDF, ask questions, and get answers based on the document's content. Powered by Gemini (`{GEMINI_MODEL_NAME}`), LangChain, FAISS, & Streamlit.")

with st.sidebar:
    if API_KEY_LOADED_FROM: st.info(f"🔑 API Key loaded from {API_KEY_LOADED_FROM}.")
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    process_button = st.button("Process Document")
    st.markdown("---")
    update_token_display()
    st.markdown("---")
    st.info(f"""
    **LLM:** {GEMINI_MODEL_NAME}
    **Embeddings:** {EMBEDDING_MODEL_NAME}
    **DB:** FAISS (In-Memory)
    """)

# --- Retryable functions for embedding and FAISS operations ---
@retry(
    wait=wait_random_exponential(min=15, max=120), # Increased min/max wait times
    stop=stop_after_attempt(7), # Increased attempts
    retry=retry_if_exception_type((
        google_api_exceptions.DeadlineExceeded,
        google_api_exceptions.ServiceUnavailable,
        google_api_exceptions.ResourceExhausted, # For 429 errors
        google_api_exceptions.Aborted, # Another possible transient error
        google_api_exceptions.InternalServerError, # 500 errors from Google
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING) # Log before sleeping on retry
)
def initialize_faiss_with_retry(initial_texts, embedding_function):
    logger.info(f"Attempting to initialize FAISS with {len(initial_texts)} chunk(s).")
    if initial_texts: # Log content of the first chunk being sent
        logger.info(f"Content of first chunk for initial FAISS (first 100 chars): '{initial_texts[0][:100]}...'")
    return FAISS.from_texts(texts=initial_texts, embedding=embedding_function)

@retry(
    wait=wait_random_exponential(min=15, max=120),
    stop=stop_after_attempt(7),
    retry=retry_if_exception_type((
        google_api_exceptions.DeadlineExceeded,
        google_api_exceptions.ServiceUnavailable,
        google_api_exceptions.ResourceExhausted,
        google_api_exceptions.Aborted,
        google_api_exceptions.InternalServerError,
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def embed_and_add_to_faiss_store(faiss_store, text_batch):
    logger.info(f"Attempting to add batch of {len(text_batch)} chunks to FAISS.")
    if text_batch:
        logger.info(f"Content of first chunk in current batch (first 100 chars): '{text_batch[0][:100]}...'")
    faiss_store.add_texts(texts=text_batch)

# --- Main Processing Logic ---
if process_button and uploaded_file is not None:
    with st.spinner("Processing document... This may take a few moments."):
        try:
            file_bytes = BytesIO(uploaded_file.read())
            st.info(f"📄 Processing PDF: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            logger.info(f"Processing PDF: {uploaded_file.name}")

            raw_text = extract_text_from_pdf(file_bytes)
            if not raw_text:
                st.error("Failed to extract text from PDF. The PDF might be empty, image-based, or corrupted.")
                logger.error("Failed to extract text from PDF.")
                st.stop()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            texts = text_splitter.split_text(raw_text)
            if not texts:
                st.error("Failed to split document text. The document might be empty.")
                logger.error("Failed to split document text into chunks.")
                st.stop()
            st.write(f"Document split into {len(texts)} chunks.")
            logger.info(f"Document split into {len(texts)} chunks.")

            embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL_NAME,
                google_api_key=GOOGLE_API_KEY,
                task_type="RETRIEVAL_DOCUMENT",
            )

            st.write("Creating vector store...")
            logger.info("Creating vector store...")
            
            # Drastically reduce initial setup batch size
            INITIAL_FAISS_SETUP_BATCH_SIZE = min(2, len(texts)) # Try with just 1 or 2 chunks initially
            initial_setup_texts = texts[:INITIAL_FAISS_SETUP_BATCH_SIZE]
            remaining_texts_for_addition = texts[INITIAL_FAISS_SETUP_BATCH_SIZE:]

            if not initial_setup_texts:
                 st.error("No text chunks available to initialize vector store (even for a minimal batch).")
                 logger.error("No text chunks for initial FAISS setup.")
                 st.stop()
            
            logger.info(f"First {len(initial_setup_texts)} chunks for initial setup: {initial_setup_texts}")
            st.session_state.vector_store = initialize_faiss_with_retry(initial_setup_texts, embeddings)
            st.success(f"FAISS index base initialized with the first {len(initial_setup_texts)} chunk(s).")
            logger.info("FAISS index base initialized.")

            FAISS_ADD_BATCH_SIZE = 20 # Batch size for adding remaining texts

            if remaining_texts_for_addition:
                st.write(f"Adding remaining {len(remaining_texts_for_addition)} chunks to FAISS in batches...")
                logger.info(f"Adding remaining {len(remaining_texts_for_addition)} chunks to FAISS...")
                num_add_batches = (len(remaining_texts_for_addition) + FAISS_ADD_BATCH_SIZE - 1) // FAISS_ADD_BATCH_SIZE
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i in range(num_add_batches):
                    batch_start_idx = i * FAISS_ADD_BATCH_SIZE
                    batch_end_idx = batch_start_idx + FAISS_ADD_BATCH_SIZE
                    current_text_batch = remaining_texts_for_addition[batch_start_idx:batch_end_idx]

                    if not current_text_batch: continue
                    
                    status_text.text(f"Adding batch {i+1} of {num_add_batches} ({len(current_text_batch)} chunks) to FAISS...")
                    try:
                        embed_and_add_to_faiss_store(st.session_state.vector_store, current_text_batch)
                    except Exception as e_batch:
                        st.error(f"Failed to embed and add batch {i+1} after multiple retries: {e_batch}")
                        st.warning("Document processing may be incomplete. Try a smaller document or check API status.")
                        logger.error(f"Failed to embed and add batch {i+1}: {e_batch}\n{traceback.format_exc()}")
                        break 
                    progress_bar.progress((i + 1) / num_add_batches)
                
                status_text.text("All batches added to FAISS.")
                progress_bar.empty()
                logger.info("Finished adding all batches to FAISS.")

            if not st.session_state.vector_store:
                st.error("Vector store creation failed. Please check logs or try a different document.")
                logger.error("Vector store is None after processing.")
                st.stop()

            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
            llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_NAME,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3,
                convert_system_message_to_human=True
            )
            st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=retriever, memory=memory, return_source_documents=True, output_key='answer'
            )

            st.session_state.processing_done = True
            st.session_state.chat_history_display = []
            st.session_state.cumulative_input_tokens = 0
            st.session_state.cumulative_output_tokens = 0
            update_token_display()
            st.success(f"✅ Document '{uploaded_file.name}' processed successfully! Ready for questions.")
            logger.info("Document processing completed successfully.")

        except Exception as e_main:
            st.error(f"An critical error occurred during document processing: {e_main}")
            st.error(f"Full traceback is available in the application logs.") # Avoid showing full traceback in UI
            logger.error(f"Critical error during document processing: {e_main}\n{traceback.format_exc()}")
            st.session_state.processing_done = False

# --- Chat Display and Input Logic ---
chat_container = st.container()
with chat_container:
    if st.session_state.processing_done:
        for message in st.session_state.chat_history_display:
            with st.chat_message(message.type):
                 st.markdown(message.content)
                 if isinstance(message, AIMessage) and hasattr(message, 'source_docs') and message.source_docs:
                     with st.expander("Show Retrieved Context Used"):
                        for i_doc, doc in enumerate(message.source_docs):
                            source_info = f"Chunk from '{doc.metadata.get('source', uploaded_file.name if uploaded_file else 'document')}'"
                            # Page numbers are not reliably added by PyPDF2 and RecursiveCharacterTextSplitter
                            # if 'page' in doc.metadata:
                            #    source_info += f", Page {doc.metadata['page']}"
                            st.markdown(f"**{source_info}**")
                            st.caption(doc.page_content)
                            st.markdown("---")
    elif not uploaded_file and not process_button: # Show only if no file processing has started
        st.info("Please upload a PDF document using the sidebar to begin.")


if st.session_state.processing_done:
    user_question = st.chat_input("Ask a question about the document...")
    if user_question:
        if st.session_state.conversation_chain:
            st.session_state.chat_history_display.append(HumanMessage(content=user_question))
            with chat_container:
                with st.chat_message("human"): st.markdown(user_question)

            with st.spinner("Thinking..."):
                try:
                    logger.info(f"Invoking conversation chain with question: {user_question[:100]}...")
                    result = st.session_state.conversation_chain({"question": user_question})
                    answer = result['answer']
                    retrieved_docs = result['source_documents']
                    logger.info(f"Received answer: {answer[:100]}...")
                    st.session_state.cumulative_input_tokens += count_tokens(user_question)
                    st.session_state.cumulative_output_tokens += count_tokens(answer)
                    update_token_display()
                    ai_message = AIMessage(content=answer)
                    ai_message.source_docs = retrieved_docs
                    st.session_state.chat_history_display.append(ai_message)
                    st.rerun()
                except Exception as e_chat:
                    st.error(f"An error occurred while getting the answer: {e_chat}")
                    st.error(f"Full traceback is available in the application logs.")
                    logger.error(f"Error in conversation chain: {e_chat}\n{traceback.format_exc()}")
                    if st.session_state.chat_history_display and isinstance(st.session_state.chat_history_display[-1], HumanMessage):
                       st.session_state.chat_history_display.pop()
        else:
            st.warning("Conversation chain not initialized. Please process a document first.")
            logger.warning("User asked question but conversation chain is not initialized.")
