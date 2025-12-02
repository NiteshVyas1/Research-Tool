# import os
# import streamlit as st
# import pickle
# import time
# from langchain import OpenAI
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS

# from dotenv import load_dotenv
# load_dotenv()  # take environment variables from .env (especially openai api key)

# st.title("RockyBot: News Research Tool 📈")
# st.sidebar.title("News Article URLs")

# urls = []
# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     urls.append(url)

# process_url_clicked = st.sidebar.button("Process URLs")
# file_path = "faiss_store_openai.pkl"

# main_placeholder = st.empty()
# llm = OpenAI(temperature=0.9, max_tokens=500)

# if process_url_clicked:
#     # load data
#     loader = UnstructuredURLLoader(urls=urls)
#     main_placeholder.text("Data Loading...Started...✅✅✅")
#     data = loader.load()
#     # split data
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=['\n\n', '\n', '.', ','],
#         chunk_size=1000
#     )
#     main_placeholder.text("Text Splitter...Started...✅✅✅")
#     docs = text_splitter.split_documents(data)
#     # create embeddings and save it to FAISS index
#     embeddings = OpenAIEmbeddings()
#     vectorstore_openai = FAISS.from_documents(docs, embeddings)
#     main_placeholder.text("Embedding Vector Started Building...✅✅✅")
#     time.sleep(2)

#     # Save the FAISS index to a pickle file
#     with open(file_path, "wb") as f:
#         pickle.dump(vectorstore_openai, f)

# query = main_placeholder.text_input("Question: ")
# if query:
#     if os.path.exists(file_path):
#         with open(file_path, "rb") as f:
#             vectorstore = pickle.load(f)
#             chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
#             result = chain({"question": query}, return_only_outputs=True)
#             # result will be a dictionary of this format --> {"answer": "", "sources": [] }
#             st.header("Answer")
#             st.write(result["answer"])

#             # Display sources, if available
#             sources = result.get("sources", "")
#             if sources:
#                 st.subheader("Sources:")
#                 sources_list = sources.split("\n")  # Split the sources by newline
#                 for source in sources_list:
#                     st.write(source)


import os
import streamlit as st
import pickle

from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv()


# ---------------- UI -----------------
st.title("📰 RockyBot: Local AI News Research Tool 📈")
st.sidebar.title("Add News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url)

process_url_clicked = st.sidebar.button("📌 Process URLs")
file_path = "faiss_store_local.pkl"

main_placeholder = st.empty()


# ---------------- Local LLM (Ollama) -----------------
llm = Ollama(model="llama3.2")  # or "mistral"


# ---------------- PROCESS URLS -----------------
if process_url_clicked:
    try:
        if not urls:
            st.warning("⚠️ Please enter at least one valid URL.")
        else:
            main_placeholder.text("🔄 Loading webpage content...")
            loader = WebBaseLoader(urls)
            data = loader.load()

            main_placeholder.text("📄 Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
            docs = text_splitter.split_documents(data)

            main_placeholder.text("🧠 Creating Embeddings...")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            main_placeholder.text("⚙️ Building vector database...")
            vectorstore = FAISS.from_documents(docs, embeddings)

            with open(file_path, "wb") as f:
                pickle.dump(vectorstore, f)

            main_placeholder.success("🎉 URLs processed and saved! Now ask a question below 👇")

    except Exception as e:
        st.error(f"❌ Error: {e}")



# ---------------- ASK QUESTION -----------------
query = st.text_input("🔍 Ask a question based on the processed news:")

if query:

    if not os.path.exists(file_path):
        st.error("⚠️ Please process URLs first.")
    else:
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever()

        prompt = ChatPromptTemplate.from_template(
            """
            You are an intelligent assistant analyzing news articles.

            Use ONLY the context provided. If you cannot find the answer, reply:
            "Not found in provided articles."

            --- CONTEXT ---
            {context}

            --- QUESTION ---
            {question}

            Answer clearly in bullet points.
            """
        )

        chain = (
            {
                "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.spinner("🧠 Thinking..."):
            answer = chain.invoke(query)

        st.header("💡 Answer:")
        st.write(answer)
