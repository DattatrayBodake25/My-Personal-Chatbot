# Import necessary modules and classes from langchain, streamlit, and utils
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st

# Display balloons using Streamlit to celebrate success
st.balloons()

# Import the 'message' function from streamlit_chat and utility functions from utils
from streamlit_chat import message
from utils import *

# Set up Streamlit app with a subheader for an AI-Powered PDF Chatbot
st.subheader("AI-Powered PDF Chatbot")

# Initialize session state to store responses and requests
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Initialize ChatOpenAI with GPT-3.5-turbo model and API key
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="my_open_ai_api_key")

# Set up conversation memory using a buffer window
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Define message templates for system and human prompts
system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Answer the question as truthfully as possible using the provided context, 
    and if the answer is not contained within the text below, say 'I don't know'"""
)
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

# Create a prompt template using message templates
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# Set up ConversationChain with defined memory, prompt template, language model, and verbosity
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# Create containers for chat history and text box
response_container = st.container()
textcontainer = st.container()

# Process user input
with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            # Get conversation string and refine the user's query
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            
            # Find context match for the refined query and predict the response
            context = find_match(refined_query)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        
        # Update session state with user's request and chatbot's response
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

# Display chat history
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')