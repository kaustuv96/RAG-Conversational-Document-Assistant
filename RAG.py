#!/usr/bin/env python
# coding: utf-8

# START OF SQLITE PATCH - This should be at the very top of the file
import sys
import importlib.util

# Only apply the patch if pysqlite3 is available
if importlib.util.find_spec("pysqlite3"):
    sys.modules["sqlite3"] = __import__("pysqlite3")
# END OF SQLITE PATCH

# Standard Libraries
import os
from io import BytesIO
import traceback # For detailed error logging

# Streamlit
import streamlit as st

# --- Global Google AI Configuration (BEFORE main function) ---
try:
    import google.generativeai as genai

    GOOGLE_API_KEY_FOR_CONFIG = os.getenv("GOOGLE_API_KEY")
    if GOOGLE_API_KEY_FOR_CONFIG:
        genai.configure(
            api_key=GOOGLE_API_KEY_FOR_CONFIG,
            transport='rest',
            request_options={"timeout": 300.0} # Timeout for REST transport
        )
        # print("[INFO] Applied global Google AI configuration.") # Optional: for server logs only
    # else:
        # print("[WARN] Google API Key not found for global config.") # Optional: for server logs only
except Exception as e_global_config:
    print(f"[ERROR] Could not apply global Google AI config: {e_global_config}") # For server logs


try:
    import chromadb
    import langchain
    # import langchain_google_genai # Not directly used, but langchain_google_genai.GoogleGenerativeAIEmbeddings is
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain.vectorstores import Chroma
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import HumanMessage, AIMessage
    import PyPDF2
    from transformers import AutoTokenizer, logging as hf_logging

    hf_logging.set_verbosity_error()
    IMPORT_SUCCESS = True
except Exception as e:
    # This error will be visible in the Streamlit app UI if imports fail
    st.error(f"Failed to import required libraries: {e}")
    IMPORT_SUCCESS = False

# --- Configuration ---
GEMINI_MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

def main():
    st.set_page_config(page_title="Conversational Document Q&A (RAG)", layout="wide")
    st.title("💬 Conversational AI Document Q&A with RAG")
    st.markdown(f"Upload a PDF, ask questions, and get answers based on the document's content. Powered by Gemini (`{GEMINI_MODEL_NAME}`), LangChain, Chroma, & Streamlit.")

    if not IMPORT_SUCCESS:
        st.error("Application failed to initialize due to missing dependencies. Please check the server logs.")
        return
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.sidebar.error("🔴 Google API Key is missing!")
        with st.sidebar.expander("How to add your API key"):
            st.markdown("""
            You have two options:
            1. **For local development**: Use `.streamlit/secrets.toml` with content: `GOOGLE_API_KEY = "your-key-here"`
            2. **For Streamlit Cloud**: Add the key in the app settings under 'Secrets'
            """)
        st.stop()

    try:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", legacy=False)
    except Exception as e:
        st.warning(f"⚠️ Failed to load tokenizer: {e}. Token counts will be estimated based on word count.")
        tokenizer = None

    def count_tokens(text: str) -> int:
        if not text: return 0
        if tokenizer:
            try: return len(tokenizer.encode(text))
            except: return len(text.split()) # Fallback
        else: return len(text.split())

    def extract_text_from_pdf(pdf_file_bytes):
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file_bytes)
            text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages if page.extract_text())
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

    # --- Sidebar ---
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        process_button = st.button("Process Document")
        st.markdown("---")
        update_token_display()
        st.markdown("---")
        st.info(f"""
        **Model:** {GEMINI_MODEL_NAME}
        **Embeddings:** {EMBEDDING_MODEL_NAME}
        **DB:** Chroma (In-Memory)
        """)

    # --- Document Processing Logic ---
    if process_button and uploaded_file is not None:
        with st.spinner("Processing document... This may take a few moments."):
            try:
                file_bytes = BytesIO(uploaded_file.read())
                st.info(f"📄 Extracting text from PDF: {uploaded_file.name}") # User feedback
                raw_text = extract_text_from_pdf(file_bytes)
                if not raw_text:
                    st.error("Failed to extract text from PDF.")
                    st.stop()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100,
                    length_function=len
                )
                texts = text_splitter.split_text(raw_text)
                if not texts:
                    st.error("Failed to split document text.")
                    st.stop()

                embeddings = GoogleGenerativeAIEmbeddings(
                    model=EMBEDDING_MODEL_NAME,
                    google_api_key=GOOGLE_API_KEY
                )
                
                # --- MANUAL BATCHING FOR CHROMA ---
                # Initialize an empty Chroma vector store first
                vector_store_temp = Chroma(
                    embedding_function=embeddings
                )
                batch_size = 5 # Number of text chunks to embed in one API call
                num_texts = len(texts)

                # Optional: give user a bit more feedback during longer batch processing
                # st.info(f"Preparing to embed {num_texts} text chunks in batches...")

                for i in range(0, num_texts, batch_size):
                    batch_texts = texts[i:i + batch_size]
                    # Minimal logging for batches, can be commented out if too verbose
                    # print(f"Embedding batch {i//batch_size + 1}...") # For server logs
                    try:
                        vector_store_temp.add_texts(texts=batch_texts)
                    except Exception as e_batch:
                        st.error(f"Error processing document (batch {i//batch_size + 1}): {e_batch}")
                        st.error(f"Batch Traceback: {traceback.format_exc()}")
                        st.stop() # Stop if a batch fails
                
                st.session_state.vector_store = vector_store_temp
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
                # --- END MANUAL BATCHING ---

                memory = ConversationBufferMemory(
                    memory_key='chat_history',
                    return_messages=True,
                    output_key='answer'
                )
                
                llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL_NAME,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.3,
                    convert_system_message_to_human=True
                )

                st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=memory,
                    return_source_documents=True,
                    output_key='answer'
                )

                st.session_state.processing_done = True
                st.session_state.chat_history_display = [] # Reset chat on new doc
                st.session_state.cumulative_input_tokens = 0 # Reset tokens
                st.session_state.cumulative_output_tokens = 0
                update_token_display()
                st.success(f"✅ Document '{uploaded_file.name}' processed successfully! Ready for questions.")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.error(f"Traceback: {traceback.format_exc()}")
                st.session_state.processing_done = False

    # --- Display Chat History ---
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
                                st.markdown(f"**Source:** `{source}`")
                                st.caption(doc.page_content)
                                if i < len(message.source_docs) - 1:
                                    st.markdown("---")

    # --- Chat Input Logic ---
    if st.session_state.processing_done:
        user_question = st.chat_input("Ask a question about the document...")

        if user_question:
            if st.session_state.conversation_chain:
                st.session_state.chat_history_display.append(HumanMessage(content=user_question))
                
                with st.chat_message("human"): # Display human message immediately
                    st.markdown(user_question)

                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.conversation_chain({"question": user_question})
                        answer = result['answer']
                        retrieved_docs = result.get('source_documents', [])

                        st.session_state.cumulative_input_tokens += count_tokens(user_question)
                        st.session_state.cumulative_output_tokens += count_tokens(answer)
                        update_token_display()

                        ai_message = AIMessage(content=answer)
                        ai_message.source_docs = retrieved_docs
                        st.session_state.chat_history_display.append(ai_message)
                        
                        # Rerun to update the chat display with the new AI message
                        st.rerun()

                    except Exception as e:
                        st.error(f"An error occurred while getting the answer: {e}")
                        st.error(f"Traceback: {traceback.format_exc()}")
                        # Optionally remove the last human message if AI response failed
                        if st.session_state.chat_history_display and \
                           isinstance(st.session_state.chat_history_display[-1], HumanMessage):
                           st.session_state.chat_history_display.pop()
            else:
                st.warning("Conversation chain not initialized. Please process a document first.")
    elif not uploaded_file and not st.session_state.processing_done :
        st.info("Please upload a PDF document using the sidebar to begin.")

if __name__ == "__main__":
    main()
