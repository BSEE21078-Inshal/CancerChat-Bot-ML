import streamlit as st
from rag_backend import load_and_index_pdf, get_rag_chain
import os

st.set_page_config(page_title="Breast Cancer Chatbot", layout="wide")
st.title("🎗️ Breast Cancer Support Chatbot")
st.markdown("Ask any question related to breast cancer, based on your uploaded documents.")

uploaded_file = st.file_uploader("Upload Breast Cancer PDF", type="pdf")

if uploaded_file is not None:
    if not os.path.exists("documents"):
        os.makedirs("documents")
    pdf_path = os.path.join("documents", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"{uploaded_file.name} uploaded successfully!")
    
    with st.spinner("Indexing PDF..."):
        load_and_index_pdf(pdf_path)
    st.success("Document indexed!")

if os.path.exists("faiss_index"):
    st.subheader("Ask a Question")
    user_input = st.text_input("Enter your question:")
    if user_input:
        chain = get_rag_chain()
        with st.spinner("Thinking..."):
            result = chain.run(user_input)
        st.markdown("**Answer:**")
        st.write(result)
