import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

st.header("Indian Law Chatbot")

OPENAI_API_KEY = "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"
#Upload your PDF file
with st.sidebar:
    st.title("Welcome to Indian Law Chatbot")
    file = st.file_uploader("Upload A PDF file and start asking your questions", type=["pdf"])

#Extract the PDF text and Break it into Chunks
if file is not None:
    text = ""
    pdf_reader = PdfReader(file)
    for index, page in enumerate(pdf_reader.pages):
        text += page.extract_text()

    # Split the text using LangChain's RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Optional: display number of chunks and preview the first few
    # st.write(f"✅ Split into {len(chunks)} chunks")
    # st.write(chunks[:3])  # show first 3 chunks

    # creating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Store chunks and embeddings in vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # chain -> take the question, get relevant document, pass it to the LLM, generate the output
    user_question = st.text_input("Enter your question", key=f"question_input_{1}")

    if user_question:
        match = vector_store.similarity_search(user_question)
        # st.write(match)

        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-4-turbo"
        )

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)

        st.write(response)
