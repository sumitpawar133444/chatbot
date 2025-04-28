from fastapi import FastAPI
from pydantic import BaseModel

# Assume `chatbot` is the compiled LangGraph app from previous steps
from chatbot_graph import chatbot

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    result = chatbot.invoke({"user_message": request.message})
    return {"response": result["bot_response"]}
