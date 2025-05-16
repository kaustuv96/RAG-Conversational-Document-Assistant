import os
import streamlit as st
# import langchain # langchain is a namespace, not typically imported directly like this. Specific modules are used.
import PyPDF2
from io import BytesIO
import time # For progress display

# LangChain specific imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate # Not used in your current code, can be removed if not planned
from langchain.schema import HumanMessage, AIMessage

# Tokenizer for Estimation
from transformers import AutoTokenizer, logging as hf_logging
hf_logging.set_verbosity_error()

# For retries
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from google.api_core import exceptions as google_api_exceptions # For specific Google API error types
import traceback # For more detailed error logging if needed



# #### Initialization ####

# --- Configuration ---
# GEMINI_MODEL_NAME = "gemini-2.0-flash" # LIKELY TYPO - see note below
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # Use a valid model, e.g., "gemini-1.5-flash-latest" or "gemini-pro"
EMBEDDING_MODEL_NAME = "models/embedding-004"

# --- API Key Handling (Slightly enhanced for Streamlit Cloud) ---
try:
    # Prefer Streamlit secrets if available (common for deployed apps)
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.info("🔑 API Key loaded from Streamlit secrets.")
except (KeyError, FileNotFoundError):
    # Fallback to environment variable (common for local development)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        st.sidebar.info("🔑 API Key loaded from environment variable.")

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

# Configure the genai library (optional if only using LangChain integrations, but good practice)
try:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
except ImportError:
    st.warning("`google-generativeai` library not found. This is okay if only using LangChain.")
except Exception as e:
    st.warning(f"Could not configure `google.generativeai` directly: {e}")


# --- Initialize Tokenizer ---
# ... (your existing tokenizer and count_tokens function are fine) ...
try:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", legacy=False)
except Exception as e:
    st.warning(f"⚠️ Failed to load tokenizer: {e}. Token counts will be estimated based on word count.")
    tokenizer = None

def count_tokens(text: str) -> int:
    if not text: return 0
    if tokenizer:
        try: return len(tokenizer.encode(text))
        except Exception: return len(text.split()) # Fallback
    else: return len(text.split()) # Fallback

# --- Helper Functions ---
# ... (your existing extract_text_from_pdf function is fine) ...
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
# ... (your existing session state is fine) ...
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None
if 'chat_history_display' not in st.session_state:
    st.session_state.chat_history_display = []
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'cumulative_input_tokens' not in st.session_state:
    st.session_state.cumulative_input_tokens = 0
if 'cumulative_output_tokens' not in st.session_state:
    st.session_state.cumulative_output_tokens = 0

def update_token_display():
    st.sidebar.markdown("### Token Usage (Cumulative Estimate)")
    st.sidebar.markdown(f"""
    **Input Tokens (Questions):** {st.session_state.cumulative_input_tokens:,}
    **Output Tokens (Answers):** {st.session_state.cumulative_output_tokens:,}
    **Total Tokens (Q+A):** {st.session_state.cumulative_input_tokens + st.session_state.cumulative_output_tokens:,}
    """)
    st.sidebar.caption("Note: Input tokens estimate user questions only, not the full context sent to the LLM.")


# ### Streamlit App Layout ---
# ... (your existing layout is fine) ...
st.set_page_config(page_title="Conversational Document Q&A (RAG)", layout="wide")
st.title("💬 Conversational AI Document Q&A with RAG")
st.markdown(f"Upload a PDF, ask questions, and get answers based on the document's content. Powered by Gemini (`{GEMINI_MODEL_NAME}`), LangChain, FAISS, & Streamlit.")

with st.sidebar:
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

# #### Processing Logic ####
# --- Main Area ---
if process_button and uploaded_file is not None:
    with st.spinner("Processing document... This may take a few moments."):
        try:
            file_bytes = BytesIO(uploaded_file.read())
            st.info(f"📄 Processing PDF: {uploaded_file.name}")

            # a) Extract Text
            raw_text = extract_text_from_pdf(file_bytes)
            if not raw_text:
                st.error("Failed to extract text from PDF. The PDF might be empty, image-based, or corrupted.")
                st.stop()

            # b) Split Text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,       # Slightly larger chunks might reduce total number of embeddings
                chunk_overlap=200,      # Adjust overlap accordingly
                length_function=len
            )
            texts = text_splitter.split_text(raw_text)
            if not texts:
                st.error("Failed to split document text into chunks. The document might be too short or an issue with splitting.")
                st.stop()
            st.write(f"Document split into {len(texts)} chunks.")

            # c) Create Embeddings Instance
            embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL_NAME,
                google_api_key=GOOGLE_API_KEY,
                task_type="RETRIEVAL_DOCUMENT", # Explicitly set for document chunks
                # request_timeout=120 # Optional: Increase default timeout for each embedding batch call (default 60s)
            )

            # d) Create Vector Store (FAISS) with Batching and Retries
            st.write("Creating vector store... This can take time for large documents.")
            
            # Define a retryable function for adding texts to FAISS
            # This will be used for adding batches of texts to the FAISS index.
            @retry(
                wait=wait_random_exponential(min=5, max=90), # Wait 5s, then 10s, etc. up to 90s between retries
                stop=stop_after_attempt(4), # Attempt up to 4 times (1 initial + 3 retries)
                retry=retry_if_exception_type((
                    google_api_exceptions.DeadlineExceeded, # For 504 errors
                    google_api_exceptions.ServiceUnavailable, # For 503 errors
                    google_api_exceptions.ResourceExhausted, # For 429 errors (rate limits)
                    # You might want to add other specific transient error types if you encounter them
                ))
            )
            def embed_and_add_to_faiss(faiss_store, text_batch):
                faiss_store.add_texts(texts=text_batch) # Langchain's FAISS uses the embedding model it was initialized with

            FAISS_BATCH_SIZE = 50 # Number of text chunks to process in each FAISS `add_texts` call
                                 # Langchain's GoogleGenerativeAIEmbeddings internally batches API calls too (e.g., up to 100 docs for Gemini)
                                 # So, FAISS_BATCH_SIZE=50 means roughly 1 Gemini API call per FAISS batch.

            # Initialize FAISS with the first batch
            first_batch_texts = texts[:FAISS_BATCH_SIZE]
            remaining_texts = texts[FAISS_BATCH_SIZE:]

            if not first_batch_texts:
                st.error("No text chunks available to initialize vector store.")
                st.stop()
            
            try:
                st.write(f"Initializing vector store with first {len(first_batch_texts)} chunk(s)...")
                # Initialize FAISS. This will make at least one call to the embedding API.
                st.session_state.vector_store = FAISS.from_texts(
                    texts=first_batch_texts,
                    embedding=embeddings
                )
            except Exception as e:
                st.error(f"Error initializing FAISS vector store with the first batch: {e}")
                st.error("This could be due to an API issue, invalid API key, or problem with the first text chunks.")
                # st.code(traceback.format_exc()) # For detailed debugging
                st.stop()

            # Add remaining texts in batches with progress
            if remaining_texts:
                num_remaining_batches = (len(remaining_texts) + FAISS_BATCH_SIZE - 1) // FAISS_BATCH_SIZE
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i in range(num_remaining_batches):
                    batch_start_idx = i * FAISS_BATCH_SIZE
                    batch_end_idx = batch_start_idx + FAISS_BATCH_SIZE
                    current_text_batch = remaining_texts[batch_start_idx:batch_end_idx]

                    if not current_text_batch:
                        continue
                    
                    status_text.text(f"Embedding batch {i+1} of {num_remaining_batches} ({len(current_text_batch)} chunks)...")
                    try:
                        # The @retry decorator will handle retries for this function call
                        embed_and_add_to_faiss(st.session_state.vector_store, current_text_batch)
                    except Exception as e:
                        st.error(f"Failed to embed and add batch {i+1} after multiple retries: {e}")
                        st.warning("Document processing may be incomplete. Try a smaller document or check API status.")
                        # st.code(traceback.format_exc())
                        break # Stop processing further batches if one fails persistently
                    
                    progress_bar.progress((i + 1) / num_remaining_batches)
                
                status_text.text("All batches processed.")
                progress_bar.empty()

            if not st.session_state.vector_store:
                st.error("Vector store creation failed. Please check logs or try a different document.")
                st.stop()

            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})

            # e) Create Memory
            memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True,
                output_key='answer'
            )

            # f) Create LLM
            llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_NAME,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3,
                convert_system_message_to_human=True
            )

            # g) Create Conversational Retrieval Chain
            st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                output_key='answer'
            )

            st.session_state.processing_done = True
            st.session_state.chat_history_display = []
            st.session_state.cumulative_input_tokens = 0
            st.session_state.cumulative_output_tokens = 0
            update_token_display()
            st.success(f"✅ Document '{uploaded_file.name}' processed successfully! Ready for questions.")

        except Exception as e:
            st.error(f"An critical error occurred during document processing: {e}")
            # st.code(traceback.format_exc()) # For detailed debugging
            st.session_state.processing_done = False


# --- Chat Display and Input Logic ---
# ... (Your existing chat display and input logic is fine) ...
# (Make sure this part is correctly indented to be outside the `if process_button...` block)

# Display Chat History
chat_container = st.container()
with chat_container:
    if st.session_state.processing_done:
        for message in st.session_state.chat_history_display:
            with st.chat_message(message.type):
                 st.markdown(message.content)
                 if isinstance(message, AIMessage) and hasattr(message, 'source_docs') and message.source_docs:
                     with st.expander("Show Retrieved Context Used"):
                        for i, doc in enumerate(message.source_docs):
                            source = doc.metadata.get('source', f'Chunk {i+1}') # Fallback source name
                            st.markdown(f"**Source:** `{source}` (Page: {doc.metadata.get('page', 'N/A')})") # If page metadata exists
                            st.caption(doc.page_content)
                            st.markdown("---")
    elif not uploaded_file and not process_button: # Initial state before any upload
        st.info("Please upload a PDF document using the sidebar to begin.")


# Chat Input Logic
if st.session_state.processing_done:
    user_question = st.chat_input("Ask a question about the document...")

    if user_question:
        if st.session_state.conversation_chain:
            st.session_state.chat_history_display.append(HumanMessage(content=user_question))
            with chat_container: # Ensure messages are added to the right container
                with st.chat_message("human"):
                    st.markdown(user_question)

            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.conversation_chain({
                        "question": user_question,
                    })
                    answer = result['answer']
                    retrieved_docs = result['source_documents']

                    st.session_state.cumulative_input_tokens += count_tokens(user_question)
                    st.session_state.cumulative_output_tokens += count_tokens(answer)
                    update_token_display()

                    ai_message = AIMessage(content=answer)
                    ai_message.source_docs = retrieved_docs
                    st.session_state.chat_history_display.append(ai_message)
                    
                    st.rerun() # Rerun to update the display with AI message and context

                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
                    # st.code(traceback.format_exc())
                    if st.session_state.chat_history_display and isinstance(st.session_state.chat_history_display[-1], HumanMessage):
                       st.session_state.chat_history_display.pop() # Remove user question if AI failed
        else:
            st.warning("Conversation chain not initialized. Please process a document first.")
# (The `elif not uploaded_file:` check for the initial message is now better placed within the chat_container logic)
