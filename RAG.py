#!/usr/bin/env python
# coding: utf-8

# #### Objectives
# 
# 1. Develop an AI-Powered Document Q&A Agent with Conversational Memory using RAG.
# 2. Allow users to upload a PDF, ask questions, and receive answers grounded in the document.
# 3. Maintain conversation history for follow-up questions.
# 4. Use Gemini, LangChain, ChromaDB, and Streamlit.
# 5. Show retrieved context used for grounding the answer.
# 6. Handle API Key loading securely and track token usage (estimate per turn).

# #### Importing Libraries

# In[4]:

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Python Standard Libraries immediately 
import os
import io # Ensure io is imported if BytesIO is used later

# Import chromadb *immediately* after the patch and standard libs
# This ensures chromadb initializes with the patched sqlite3
import chromadb # <<<<<<<<<<< IMPORT chromadb HERE

# Now Streamlit, which might also have some early initializations
import streamlit as st

# LangChain specific imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate # Optional: for custom prompts if needed
from langchain.schema import HumanMessage, AIMessage # To structure chat history for display

# Other utilities
import PyPDF2
from io import BytesIO # Moved standard lib 'io' import earlier for clarity
from transformers import AutoTokenizer, logging as hf_logging

# Suppress tokenizer warnings if needed
hf_logging.set_verbosity_error()


# #### Tokenizer for Estimation

# In[5]:


from transformers import AutoTokenizer, logging as hf_logging
# Suppress tokenizer warnings if needed
hf_logging.set_verbosity_error()


# #### Printing the Versions of Libraries Used

# In[6]:


print("Libraries used:")
print(f"- streamlit: {st.__version__}")
print(f"- langchain: {langchain.__version__}")
print(f"- chromadb: {chromadb.__version__}")
print(f"- PyPDF2: {PyPDF2.__version__}")


# #### Initialization

# In[ ]:


# --- Configuration ---
GEMINI_MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL_NAME = "models/embedding-001" # Standard Gemini embedding model
# EMBEDDING_MODEL_NAME = "gemini-embedding-exp-03-07" # Standard Gemini embedding model

# --- API Key Handling ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("🔴 Google API Key is missing! Please set the GOOGLE_API_KEY environment variable.")
    # ... (rest of the API key instructions - same as before) ...
    st.stop()

# --- Initialize Tokenizer ---
try:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", legacy=False)
except Exception as e:
    st.warning(f"⚠️ Failed to load tokenizer: {e}. Token counts will be estimated based on word count.")
    tokenizer = None

def count_tokens(text: str) -> int:
    """Counts tokens using the loaded tokenizer or estimates with word count."""
    # ... (same count_tokens function as before) ...
    if not text: return 0
    if tokenizer:
        try: return len(tokenizer.encode(text))
        except Exception: return len(text.split())
    else: return len(text.split())

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_bytes):
    """Extracts text from PDF file bytes."""
    # ... (same extract_text_from_pdf function as before) ...
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


# #### Setting up Streamlit

# In[ ]:


# --- Streamlit Session State Initialization ---
# To store state across reruns
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None
if 'chat_history_display' not in st.session_state: # Separate history for UI display
    st.session_state.chat_history_display = []
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
# Token tracking (cumulative per session)
if 'cumulative_input_tokens' not in st.session_state:
    st.session_state.cumulative_input_tokens = 0
if 'cumulative_output_tokens' not in st.session_state:
    st.session_state.cumulative_output_tokens = 0

def update_token_display():
    """Updates cumulative token usage display."""
    st.sidebar.markdown("### Token Usage (Cumulative Estimate)")
    # Simplified: Tracks Q+A tokens, not full context/history tokens sent to LLM
    st.sidebar.markdown(f"""
    **Input Tokens (Questions):** {st.session_state.cumulative_input_tokens:,}
    **Output Tokens (Answers):** {st.session_state.cumulative_output_tokens:,}
    **Total Tokens (Q+A):** {st.session_state.cumulative_input_tokens + st.session_state.cumulative_output_tokens:,}
    """)
    st.sidebar.caption("Note: Input tokens estimate user questions only, not the full context sent to the LLM.")

# ### Streamlit App Layout ---
st.set_page_config(page_title="Conversational Document Q&A (RAG)", layout="wide")
st.title("💬 Conversational AI Document Q&A with RAG")
st.markdown(f"Upload a PDF, ask questions, and get answers based on the document's content. Powered by Gemini (`{GEMINI_MODEL_NAME}`), LangChain, Chroma, & Streamlit.")

# --- Sidebar ---
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    process_button = st.button("Process Document")
    st.markdown("---")
    # Display token counts
    update_token_display()
    st.markdown("---")
    st.info(f"""
    **Model:** {GEMINI_MODEL_NAME}
    **Embeddings:** {EMBEDDING_MODEL_NAME}
    **DB:** Chroma (In-Memory)
    """)


# #### Processing Logic

# 1. Document Processing Logic (only runs when button clicked and file uploaded)

# In[ ]:


# --- Main Area ---
if process_button and uploaded_file is not None:
    with st.spinner("Processing document... This may take a few moments."):
        try:
            file_bytes = BytesIO(uploaded_file.read())
            st.info(f"📄 Processing PDF: {uploaded_file.name}")

            # a) Extract Text
            raw_text = extract_text_from_pdf(file_bytes)
            if not raw_text:
                st.error("Failed to extract text from PDF.")
                st.stop()

            # b) Split Text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200, # Adjusted chunk size
                chunk_overlap=150, # Adjusted overlap
                length_function=len
            )
            texts = text_splitter.split_text(raw_text)
            if not texts:
                st.error("Failed to split document text.")
                st.stop()

            # c) Create Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
            
            # d) Create Vector Store (ChromaDB In-Memory)
            st.session_state.vector_store = Chroma.from_texts(
                texts=texts,
                embedding=embeddings,
                # Optional: Add source metadata if splitting preserves page numbers etc.
                # metadatas=[{"source": f"{uploaded_file.name}-chunk-{i}"} for i in range(len(texts))]
            )
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 chunks

            # e) Create Memory
            # output_key='answer' ensures memory correctly captures the AI response
            memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True,
                output_key='answer' # Crucial for ConversationalRetrievalChain
            )

            # f) Create LLM
            llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_NAME,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3, # Lower temperature for more factual Q&A
                convert_system_message_to_human=True # Often needed for Gemini compatibility
            )

            # g) Create Conversational Retrieval Chain
            st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True, # <<< Important to get sources
                output_key='answer'           # <<< Explicitly define output key
                # You can add custom prompts here if needed using `combine_docs_chain_kwargs`
            )

            st.session_state.processing_done = True
            st.session_state.chat_history_display = [] # Reset display history for new doc
            st.session_state.cumulative_input_tokens = 0 # Reset token counts
            st.session_state.cumulative_output_tokens = 0
            update_token_display() # Update sidebar counts
            st.success(f"✅ Document '{uploaded_file.name}' processed successfully! Ready for questions.")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.session_state.processing_done = False
            # print(traceback.format_exc()) # For debugging


# 2. Display Chat History

# In[ ]:


# Use st.container() for better layout control if needed
chat_container = st.container()
with chat_container:
    if st.session_state.processing_done:
        for message in st.session_state.chat_history_display:
            with st.chat_message(message.type): # 'human' or 'ai'
                 st.markdown(message.content)
                 # Display context only for AI messages where it was captured
                 if isinstance(message, AIMessage) and hasattr(message, 'source_docs'):
                     with st.expander("Show Retrieved Context Used"):
                        for doc in message.source_docs:
                            # Try to get source metadata if available
                            source = doc.metadata.get('source', 'Unknown chunk')
                            st.markdown(f"**Source:** `{source}`")
                            st.caption(doc.page_content)
                            st.markdown("---")


# 3. Chat Input Logic

# In[ ]:


if st.session_state.processing_done:
    user_question = st.chat_input("Ask a question about the document...")

    if user_question:
        if st.session_state.conversation_chain:
            # Add user question to display history immediately
            st.session_state.chat_history_display.append(HumanMessage(content=user_question))

            # Display user message in chat container
            with chat_container:
                with st.chat_message("human"):
                    st.markdown(user_question)

            # Process the question using the chain
            with st.spinner("Thinking..."):
                try:
                    # The chain uses its internal memory which includes previous turns
                    # We pass the current question and the *internal* memory handles history
                    result = st.session_state.conversation_chain({
                        "question": user_question,
                        # chat_history is implicitly handled by the memory object passed during chain creation
                    })
                    answer = result['answer']
                    retrieved_docs = result['source_documents']

                    # Update token counts (estimation)
                    st.session_state.cumulative_input_tokens += count_tokens(user_question)
                    st.session_state.cumulative_output_tokens += count_tokens(answer)
                    update_token_display()

                    # Create AI message with source docs attached for display
                    ai_message = AIMessage(content=answer)
                    ai_message.source_docs = retrieved_docs # Attach docs for the expander

                    # Add AI response to display history
                    st.session_state.chat_history_display.append(ai_message)

                    # Rerun the script to update the chat display including the new AI message and context
                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
                    # Remove the user message we optimistically added if AI fails
                    if st.session_state.chat_history_display and isinstance(st.session_state.chat_history_display[-1], HumanMessage):
                       st.session_state.chat_history_display.pop()


        else:
            st.warning("Conversation chain not initialized. Please process a document first.")
elif not uploaded_file:
    st.info("Please upload a PDF document using the sidebar to begin.")

