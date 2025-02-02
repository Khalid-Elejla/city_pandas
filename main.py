#=======================================================================================================
import os
import streamlit as st
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import io

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

data_path = "data/updated_riyadh_complaints.csv"
df = pd.read_csv(data_path)

def capture_plot():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return buf

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4"),
    df,
    verbose=False,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True,
    temperature=0
)


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    st.set_page_config(
        page_title="city link", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS to reduce spacing
    st.markdown("""
        <style>
            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
            div[data-testid="stVerticalBlock"] > div:first-child {
                padding-top: 0;
            }
            .stMarkdown h1 {
                margin: 0 0 0.5rem 0 !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    initialize_session_state()

    # Main content with reduced spacing
    st.markdown("""
        <h1 style='text-align: center; margin: 0; padding: 0;'>
            Welcome to <span style='color: blue;'>CITYLINK</span> assistant
        </h1>
    """, unsafe_allow_html=True)

    # Map with reduced top margin
    st.markdown("<div style='margin-top: 0.5rem;'>", unsafe_allow_html=True)
    st.map(df[['latitude', 'longitude']].dropna())
    st.markdown("</div>", unsafe_allow_html=True)

    # Sidebar chat
    with st.sidebar:
        st.header("Chat Assistant", divider='blue')
        
        # Chat history
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.container():
                    st.markdown(f"**You:** {message['user']}")
                    st.markdown(f"**Assistant:** {message['assistant']}")
                    if 'plot' in message and message['plot'] is not None:
                        st.image(message['plot'])
                    st.markdown("---")

        # Chat input
        user_input = st.chat_input("Type your question here...")
        if user_input:
            with st.spinner("Thinking..."):
                try:
                    response = agent.invoke(user_input)["output"]
                    plot_data = None
                    
                    if plt.get_fignums():
                        plot_data = capture_plot()

                    st.session_state.messages.append({
                        "user": user_input,
                        "assistant": response,
                        "plot": plot_data
                    })
                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()