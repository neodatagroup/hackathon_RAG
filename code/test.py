from dotenv import load_dotenv, find_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor


import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler

# Streaming Handler
class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


load_dotenv(find_dotenv())


# Load and prepare documents
loader = WebBaseLoader(
    [
        "https://web.dmi.unict.it/corsi/l-31/insegnamenti?seuid=CD1ABF9F-5308-450E-813E-60B84F9EDAA5",
        "https://web.dmi.unict.it/corsi/l-31/insegnamenti?seuid=6E03B0E2-5E93-43C5-BBFB-E4D6446DB180",
        "https://web.dmi.unict.it/corsi/l-31/insegnamenti?seuid=81E1DC57-5DC2-46ED-84AF-3C8BB46F3F49",
        "https://web.dmi.unict.it/corsi/l-31/contatti",
    ]
)
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=400
).split_documents(docs)
vectordb = Chroma.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6})

# Create tools
retriever_tool = create_retriever_tool(
    retriever,
    "Uni_helper",
    "Help students and Search for information about University of Catania courses. For any questions about uni courses and their careers, you must use this tool for helping students!",
)
# Search tool
search = TavilySearchResults(max_results=3)
tools = [search, retriever_tool]

# Initialize the model
model = ChatOpenAI(model="gpt-4o", streaming=True)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer always in the language of the question",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create the agent
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit app
st.set_page_config(page_title="Neodata Assistant", page_icon="🌐")
st.header("Your personal assistant 🤖")
st.write(
    """Hi. I am an agent powered by Neodata.
I will be your virtual assistant to help you with your experience within the student portal. 
Ask me anything about your courses or academic career"""
)
st.write(
    "[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/3_%F0%9F%8C%90_chatbot_with_internet_access.py)"
)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    st.header("I tuoi corsi")
    st.markdown("- INTRODUZIONE AL DATA MINING")
    st.markdown("- BASI DI DATI A - L")
    st.markdown("- ALGEBRA LINEARE E GEOMETRIA A - E")

    st.divider()
    
    st.markdown("Chiedimi i contatti della segreteria!")

        


if prompt := st.chat_input("What is up?", key="first_question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        stream_handler = StreamHandler(st.empty())
        # Execute the agent with chat history
        result = agent_executor(
            {
                "input": prompt,
                "chat_history": [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
            },
            callbacks=[stream_handler],
        )
        response = result.get("output")

    st.session_state.messages.append({"role": "assistant", "content": response})
    # st.chat_message("assistant").markdown(response)
