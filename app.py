import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page config
st.set_page_config(
    page_title="Science RAG System",
    page_icon="🔬",
    layout="wide"
)

# Title
st.title("🔬 Science Question Answering System")
st.markdown("Ask any basic Science question!")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.header("ℹ️ Information")
    st.success("✅ Ready to answer questions!")
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("This RAG system answers basic Science questions using Wikipedia content.")
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Get API key from Streamlit secrets (or hardcode for local testing)
try:
    api_key = st.secrets["GROQ_API_KEY"]  # For deployed version
except:
    api_key = "YOUR_GROQ_API_KEY_HERE"  # For local testing - REPLACE THIS

# Load vector database
@st.cache_resource
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vectordb = FAISS.load_local(
        folder_path="./faiss_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vectordb

# Create RAG chain
@st.cache_resource
def create_rag_chain(_vectordb, _api_key):
    llm = ChatGroq(
        api_key=_api_key,
        model="llama-3.1-8b-instant",
        temperature=0.3
    )
    
    retriever = _vectordb.as_retriever(search_kwargs={"k": 3})
    
    template = """You are a helpful Science teacher. Answer the question based on the context below.
    If you cannot answer the question based on the context, say "I don't have enough information to answer that."
    Keep your answer clear and concise.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# Main app
try:
    # Load resources
    with st.spinner("Loading knowledge base..."):
        vectordb = load_vectordb()
        rag_chain, retriever = create_rag_chain(vectordb, api_key)
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if query := st.chat_input("Ask a Science question..."):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get relevant documents - FIXED LINE
                docs = retriever.invoke(query)
                
                # Get answer
                answer = rag_chain.invoke(query)
                
                # Display answer
                st.markdown(answer)
                
                # Display sources
                with st.expander("📚 Sources"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(doc.page_content[:300] + "...")
                        st.markdown("---")
        
        # Add assistant message to chat
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Please contact the administrator if the issue persists.")

# Footer
st.markdown("---")