# Import necessary libraries
import time
import re
from langchain_core.prompts import PromptTemplate
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
import boto3
from langchain_aws import BedrockLLM as Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from pathlib import Path

st.set_page_config(page_title="VIA", page_icon="🤖")

st.header("VIA Bot")
st.success("Status: Online")

# Prompt
template = """Use the following context documents to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer.
2. If you find the answer, write the answer in a detailed way without references.
{context}
Question: {input}
Helpful Answer:"""

BUCKET_NAME = "insuranceragchatbot"
file_path = f"/tmp"

# Initialize AWS connectors
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
s3 = boto3.client(service_name="s3")

prompt = PromptTemplate(
    input_variables=["context", "input"],
    template=template,
)

# Method to use the foundational LLM model via bedrock
def get_llama2_llm():
    llm = Bedrock(model_id="meta.llama2-13b-chat-v1", client=bedrock,
                  model_kwargs={'max_gen_len': 512})
    return llm

llm = get_llama2_llm()

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Method to download vectors of the policy document from S3
def download_vectors(policy_number):
    s3_vector_faiss_key = 'vectors/policydoc/' + policy_number + '/' + 'policydoc_faiss.faiss'
    s3_vector_pkl_key = 'vectors/policydoc/' + policy_number + '/' + 'policydoc_pkl.pkl'
    Path(file_path).mkdir(parents=True, exist_ok=True)
    s3.download_file(Bucket=BUCKET_NAME, Key=s3_vector_faiss_key, Filename=f"{file_path}/my_faiss.faiss")
    s3.download_file(Bucket=BUCKET_NAME, Key=s3_vector_pkl_key, Filename=f"{file_path}/my_faiss.pkl")
   
# Method to load the vector indexes
def load_faiss_index():
    faiss_index = FAISS.load_local(index_name="my_faiss", folder_path=file_path, embeddings=bedrock_embeddings, allow_dangerous_deserialization=True)
    retriever = faiss_index.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    chain = create_retrieval_chain(retriever_chain, document_chain)
    return chain

# Methods to retreive chat responses as stream
def get_response(query, chain):
    return chain.stream({"input": query})

def get_streamed_response(prompt, chain):
    chunks = []
    for chunk in get_response(prompt, chain):
        if "answer" in chunk:
            chunks.append(chunk["answer"])
    return ''.join(chunks)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi! I'm VIA, your Virtual Insurance Assistant"),
    ]
    time.sleep(2)
    st.session_state.chat_history.append(AIMessage(content="How can I help you today?"))

# Display chat messages using Streamlit's chat_message
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)


# Handle user input
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

# Handle user input
prompt_ = st.chat_input("How can I help you?")

if prompt_:
    # Display user message in chat message container
    st.session_state.chat_history.append(HumanMessage(content=prompt_))
    st.session_state.awaiting_response = True
    st.rerun()


    user_input = st.session_state.chat_history[-1].content
    response = get_streamed_response(user_input, st.session_state.chain)

    # Add the response to chat history
    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.awaiting_response = False
    st.rerun()