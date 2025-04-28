from fastapi import FastAPI
from app.models import ChatRequest
from app.chatbot_graph import chatbot

app = FastAPI()

# In-memory chat sessions (only for demo — in production use RDS!)
session_memory = {}

@app.post("/chat/{session_id}")
async def chat(session_id: str, request: ChatRequest):
    memory = session_memory.get(session_id, [])
    result = chatbot.invoke({
        "user_message": request.message,
        "chat_history": memory
    })
    session_memory[session_id] = result["chat_history"]
    return {"response": result["bot_response"]}
