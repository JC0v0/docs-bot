import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os

os.environ["OPENAI_API_BASE"]="https://api.jingcheng.love/v1"
os.environ["OPENAI_API_KEY"]="sk-Dt0A7pMAVQMqExD76d0aA96700Ad4a3e9e46D3D2F4Db545a"
st.title("文档GPT")

#加载文档
loader = UnstructuredPDFLoader(
        "mb_manual_b760m-ty-pioneer-wifi_1001_sc.pdf", mode="single", strategy="fast",
    )
docs = loader.load()
#嵌入向量存储
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())


template = """使用以下上下文来回答最后的问题。
如果你不知道答案，就说你不知道，不要试图编造答案。
最多使用五个句子，并尽可能保持答案简洁。
总是说“谢谢你的询问！” 在答案的最后。
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    memory=memory
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入你的问题"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = qa_chain({"query": prompt})

    with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
