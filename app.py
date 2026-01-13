import streamlit as st
from agent import run_agent
from ingest import build_vector_db

# Page config
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="+",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .main-header {
        text-align: center;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Medical RAG Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Powered by LangChain + Ollama + PubMed</p>", unsafe_allow_html=True)

# Sidebar for ingesting new research
with st.sidebar:
    st.header("Add New Research")
    new_topic = st.text_input("Research Topic", placeholder="e.g., Diabetes treatment 2024")
    
    if st.button("Ingest Research", use_container_width=True):
        if new_topic:
            with st.spinner(f"Fetching papers on '{new_topic}'..."):
                try:
                    build_vector_db(new_topic)
                    st.success(f"Added research on '{new_topic}' to knowledge base!")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a topic")
    
    st.divider()
    st.markdown("### How to use")
    st.markdown("""
    1. **Ask questions** about medical topics
    2. **Add research** using the sidebar
    3. The agent decides whether to:
       - Search PubMed for new papers
       - Query existing knowledge base
       - Add new research automatically
    """)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a medical question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = run_agent(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
