import streamlit as st
from graph import create_graph
from langchain_core.messages import HumanMessage, AIMessage
from db_utils import delete_conversation

# --- App Configuration ---
st.set_page_config(
    page_title="Financial Analyst Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("🤖 Financial Analyst Assistant")
st.write(
    "Welcome! I can answer questions about the latest 10-K filings for major tech companies."
)

# --- Sidebar Controls ---
st.sidebar.title("Controls")
if st.sidebar.button("Clear Conversation History"):
    # Use the thread_id from the current session
    delete_conversation(st.session_state.thread_id)
    # Give user feedback
    st.sidebar.success("Conversation history has been cleared!")
    # Rerun the app to refresh the chat display
    st.rerun()

# --- LangGraph Agent Initialization ---
# This should be done once and cached for performance
@st.cache_resource
def get_agent_app():
    """Creates and returns the LangGraph agent application."""
    return create_graph()

agent_app = get_agent_app()

# --- Session State Management ---
# Ensure a unique thread_id for each user session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = 1 # Using a static ID for simplicity but can be replaced with username

# --- Configuration for LangGraph ---
# The thread_id is retrieved from the session state
config = {
    "configurable": {
        "thread_id": st.session_state.thread_id,
        "recursion_limit": 50
    }
}

# --- Chat History Display Function ---
def display_chat_history(app, config):
    """Displays messages from the history for the given config."""
    # Get the history from the LangGraph's checkpointer
    history = app.get_state(config)
    if not history:
        return
    
    messages = history.values.get('messages', [])
    for msg in messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

# --- Main App Logic ---

# Display the existing chat history on every page load
display_chat_history(agent_app, config)

# Get user input from the chat interface
if user_prompt := st.chat_input("Ask your question here..."):
    
    # Display the user's message immediately
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    # The user's new message is a HumanMessage
    initial_state = {
        "messages": [HumanMessage(content=user_prompt)]
    }

    # Display a spinner while the agent is processing the query
    with st.spinner("Analyzing filings and generating response..."):
        # Invoke the LangGraph agent
        agent_app.invoke(
            initial_state,
            config=config
        )

    # Rerun the Streamlit app to automatically display the new messages
    # The display_chat_history function will now include the latest turn
    st.rerun()