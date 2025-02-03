# import os
# import io
# import base64
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pandas as pd

# import matplotlib
# matplotlib.use('Agg')  # Set backend first to prevent display

# import matplotlib.pyplot as plt
# from langchain.agents.agent_types import AgentType
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# app = FastAPI()

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # CORS configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load data and initialize agent
# df = pd.read_csv("data/updated_riyadh_complaints.csv")

# agent = create_pandas_dataframe_agent(
#     ChatOpenAI(temperature=0, model="gpt-4"),
#     df,
#     verbose=False,
#     agent_type=AgentType.OPENAI_FUNCTIONS,
#     allow_dangerous_code=True,
# )

# class QueryRequest(BaseModel):
#     query: str

# def capture_plots():
#     """Capture all matplotlib plots as base64 encoded images"""
#     plot_images = []
#     for fig_num in plt.get_fignums():
#         # Switch to the figure
#         fig = plt.figure(fig_num)
#         buf = io.BytesIO()
#         fig.savefig(buf, format='png', bbox_inches='tight')
#         buf.seek(0)
#         plot_images.append(base64.b64encode(buf.read()).decode('utf-8'))
#         plt.close(fig)
#     return plot_images

# @app.post("/chat")
# async def process_query(request: QueryRequest):
#     try:
#         # Process query through LangChain agent
#         response = agent.invoke(request.query)["output"]

#         # Capture all plots if any exist
#         plot_images = capture_plots() if plt.get_fignums() else []

#         return {
#             "response": response,
#             "plot_images": plot_images
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#===============================================================================================================================

# import os
# import io
# import base64
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pandas as pd
# import uuid
# from typing import Dict, List
# from datetime import datetime, timedelta

# import matplotlib
# matplotlib.use('Agg')

# import matplotlib.pyplot as plt
# from langchain.agents.agent_types import AgentType
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# app = FastAPI()

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # CORS configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Session storage (use Redis in production)
# sessions: Dict[str, dict] = {}  # Format: {session_id: {messages: [], created_at: datetime}}

# # Session configuration
# SESSION_TTL = timedelta(minutes=30)  # 30-minute session expiry

# # Load data and initialize agent
# df = pd.read_csv("data/updated_riyadh_complaints.csv")

# agent = create_pandas_dataframe_agent(
#     ChatOpenAI(temperature=0, model="gpt-4"),
#     df,
#     verbose=False,
#     agent_type=AgentType.OPENAI_FUNCTIONS,
#     allow_dangerous_code=True,
# )

# class QueryRequest(BaseModel):
#     query: str
#     session_id: str = None  # Frontend should send this for follow-ups

# class Message(BaseModel):
#     content: str
#     is_user: bool
#     timestamp: datetime
#     plots: List[str] = []

# def capture_plots():
#     """Capture all matplotlib plots as base64 encoded images"""
#     plot_images = []
#     for fig_num in plt.get_fignums():
#         fig = plt.figure(fig_num)
#         buf = io.BytesIO()
#         fig.savefig(buf, format='png', bbox_inches='tight')
#         buf.seek(0)
#         plot_images.append(base64.b64encode(buf.read()).decode('utf-8'))
#         plt.close(fig)
#     return plot_images

# # def get_or_create_session(session_id: str) -> dict:
# #     """Manage session lifecycle"""
# #     if session_id and session_id in sessions:
# #         # Reset TTL on activity
# #         sessions[session_id]["expires_at"] = datetime.now() + SESSION_TTL
# #         return sessions[session_id]
    
# #     # Create new session
# #     new_session_id = str(uuid.uuid4())
# #     sessions[new_session_id] = {
# #         "messages": [],
# #         "created_at": datetime.now(),
# #         "expires_at": datetime.now() + SESSION_TTL
# #     }
# #     return sessions[new_session_id]

# def get_or_create_session(session_id: str) -> tuple[str, dict]:
#     """Returns (session_id, session_data)"""
#     if session_id in sessions:
#         session = sessions[session_id]
#         session["expires_at"] = datetime.now() + SESSION_TTL
#         return session_id, session
    
#     new_session_id = str(uuid.uuid4())
#     sessions[new_session_id] = {
#         "messages": [],
#         "created_at": datetime.now(),
#         "expires_at": datetime.now() + SESSION_TTL
#     }
#     return new_session_id, sessions[new_session_id]


# def cleanup_sessions():
#     """Remove expired sessions"""
#     now = datetime.now()
#     expired = [sid for sid, session in sessions.items() if session["expires_at"] < now]
#     for sid in expired:
#         del sessions[sid]

# @app.post("/chat")
# async def process_query(request: QueryRequest):
#     cleanup_sessions()  # Remove expired sessions first
    
#     try:
#         # session = get_or_create_session(request.session_id)
#         # session_id = [k for k, v in sessions.items() if v == session][0]
#         session_id, session = get_or_create_session(request.session_id)

#         # Build conversation history
#         history = "\n".join(
#             [f"{'User' if msg['is_user'] else 'AI'}: {msg['content']}" 
#              for msg in session['messages'][-10:]]  # Keep last 10 messages
#         )
        
#         # Format the query with context
#         full_query = f"Conversation history:\n{history}\n\nNew query: {request.query}"
        
#         # Process query through LangChain agent
#         response = agent.invoke(full_query)["output"]
        
#         # Capture plots
#         plot_images = capture_plots() if plt.get_fignums() else []
        
#         # Store messages
#         session['messages'].append({
#             "content": request.query,
#             "is_user": True,
#             "timestamp": datetime.now(),
#             "plots": []
#         })
        
#         session['messages'].append({
#             "content": response,
#             "is_user": False,
#             "timestamp": datetime.now(),
#             "plots": plot_images
#         })
        
#         return {
#             "response": response,
#             "plot_images": plot_images,
#             "session_id": session_id,
#             "history": session['messages'][-10:]  # Optional: send recent history
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



import os
import io
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import uuid
from typing import Dict, List
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
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

# Session storage
sessions: Dict[str, dict] = {}
SESSION_TTL = timedelta(minutes=30)

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
    session_id: str = None

class Message(BaseModel):
    content: str
    is_user: bool
    timestamp: datetime
    plots: List[str] = []

def capture_plots():
    """Capture matplotlib plots as base64 images"""
    plot_images = []
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_images.append(base64.b64encode(buf.read()).decode('utf-8'))
        plt.close(fig)
    return plot_images

def get_or_create_session(session_id: str) -> tuple[str, dict]:
    """Manage session lifecycle with proper ID handling"""
    cleanup_sessions()
    
    # Validate and use existing session
    if session_id and session_id in sessions:
        sessions[session_id]["expires_at"] = datetime.now() + SESSION_TTL
        return session_id, sessions[session_id]
    
    # Create new session with ID handling
    if session_id and isinstance(session_id, str) and len(session_id) <= 64:
        # Ensure unique ID if client-provided
        original_id = session_id
        while session_id in sessions:
            session_id = f"{original_id}_{uuid.uuid4().hex[:4]}"
    else:
        session_id = str(uuid.uuid4())
    
    sessions[session_id] = {
        "messages": [],
        "created_at": datetime.now(),
        "expires_at": datetime.now() + SESSION_TTL
    }
    return session_id, sessions[session_id]

def cleanup_sessions():
    """Remove expired sessions"""
    now = datetime.now()
    expired = [sid for sid, session in sessions.items() if session["expires_at"] < now]
    for sid in expired:
        del sessions[sid]

@app.post("/chat")
async def process_query(request: QueryRequest):
    try:
        # Get or create session
        session_id, session = get_or_create_session(request.session_id)
        
        # Build conversation history
        history_messages = session['messages'][-10:]  # Last 10 messages
        history = "\n".join(
            [f"{'User' if msg['is_user'] else 'AI'}: {msg['content']}" 
             for msg in history_messages]
        )
        
        # Process query
        full_query = f"Conversation history:\n{history}\n\nNew query: {request.query}"
        response = agent.invoke(full_query)["output"]
        plot_images = capture_plots()
        
        # Store messages
        session['messages'].append({
            "content": request.query,
            "is_user": True,
            "timestamp": datetime.now(),
            "plots": []
        })
        
        session['messages'].append({
            "content": response,
            "is_user": False,
            "timestamp": datetime.now(),
            "plots": plot_images
        })
        
        return {
            "response": response,
            "plot_images": plot_images,
            "session_id": session_id,
            "history": [Message(**msg) for msg in session['messages'][-10:]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)