import os
import io
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Set backend first to prevent display

import matplotlib.pyplot as plt
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data and initialize agent
df = pd.read_csv("data/updated_riyadh_complaints.csv")

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4"),
    df,
    verbose=False,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True,
)

class QueryRequest(BaseModel):
    query: str

def capture_plots():
    """Capture all matplotlib plots as base64 encoded images"""
    plot_images = []
    for fig_num in plt.get_fignums():
        # Switch to the figure
        fig = plt.figure(fig_num)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_images.append(base64.b64encode(buf.read()).decode('utf-8'))
        plt.close(fig)
    return plot_images

@app.post("/chat")
async def process_query(request: QueryRequest):
    try:
        # Process query through LangChain agent
        response = agent.invoke(request.query)["output"]

        # Capture all plots if any exist
        plot_images = capture_plots() if plt.get_fignums() else []

        return {
            "response": response,
            "plot_images": plot_images
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))