from fastapi import FastAPI
from models import ChatRequest
from chatbot_graph import chatbot
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    result = chatbot.invoke({"user_message": request.message})
    return {"response": result["bot_response"]}
