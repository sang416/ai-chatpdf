__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
import streamlit as st
import tempfile
import os
import chromadb.api.client
from streamlit_extras.buy_me_a_coffee import button

chromadb.api.client.Client.clear_system_cache()

# from dotenv import load_dotenv
# load_dotenv()

# Streamlit Title
st.title("Chat PDF")
st.write("---")
button(username="sang416", floating=True, width=221)

# Streamlit File Uploader
uploaded_file = st.file_uploader("Choose a file",type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file) :
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# Code after file upload
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    # Embedder
    embeddings_model = OpenAIEmbeddings()

    # Vector store
    db = Chroma.from_documents(texts, embeddings_model)
    # Question
    st.header("PDF에게 질문해보세요 :")
    question = st.text_input("질문을 입력하세요")
    
    if st.button("질문하기"):
        with st.spinner("PDF에게 질문 중입니다..."):
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm=llm,retriever=db.as_retriever())
            result = qa_chain({"query":question})
            st.write(result["result"])