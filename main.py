import os
import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_huggingface import HuggingFaceEndpoint
from tool import ImageCaptionTool
from secret_key import sec_key
from tempfile import NamedTemporaryFile
import tempfile

os.environ['HUGGINGFACEHUB_API_TOKEN'] = sec_key  # secret key used

tool = [ImageCaptionTool()]  # list having custom tool

conversational_memory = ConversationBufferWindowMemory(  # create memory buffer to remember chats
    memory_key='chat_history',
    k=5,  # remember upto 5 previous chats
    return_messages=True
)

repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'  # llm model used
llm = HuggingFaceEndpoint(  # build llm
    repo_id=repo_id,
    temperature=0.6,
    model_kwargs={"max_length": 128}
)

agent = initialize_agent(  # initialize agent
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tool,
    llm=llm,
    max_iterations=5,
    memory=conversational_memory,
    early_stoppy_method='generate',
    handle_parsing_errors=True  # Allow the agent to handle parsing errors
)

st.title('Ask a question to an image')  # title of page
st.sidebar.header("Please upload an image")  # sidebar header
file = st.sidebar.file_uploader("", type=["jpeg", "jpg", "png"])  # asks user to upload an image
if file:
    st.image(file, width=350)  # display image
    user_question = st.text_input('Ask a question about your image:')  # ask question to user
    with NamedTemporaryFile(delete=False, suffix=".png", dir=tempfile.gettempdir()) as temp_file:
        temp_file.write(file.getbuffer())
        temp_file.flush()  # Ensure the file is written completely
        image_path = temp_file.name
        if user_question and user_question != "":  # return response
            with st.spinner(text="Generating Response..."):
                response = agent.run(
                    "Given the image path '{}', provide a JSON response with 'action' and 'action_input' fields. "
                    "Ensure the response is properly formatted as JSON.".format(image_path)
                )
                st.write(response)
