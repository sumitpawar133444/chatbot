from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.chains import LLMChain
from .config import llm

# Chatbot State
class ChatState(dict):
    """
    Holds:
      - user_message: latest user message
      - chat_history: list of past user and assistant messages
    """
    pass

# Prompt Template with history
prompt = PromptTemplate(
    input_variables=["chat_history", "user_message"],
    template=(
        "You are a helpful AI assistant.\n\n"
        "Conversation so far:\n{chat_history}\n\n"
        "User: {user_message}\n\nAssistant:"
    )
)

# Simple LangChain QA Chain
qa_chain = LLMChain(llm=llm, prompt=prompt)

# Graph Nodes
def start_node(state):
    # Initialize history if not there
    if "chat_history" not in state:
        state["chat_history"] = []
    return state

def answer_node(state):
    user_message = state["user_message"]
    chat_history = state.get("chat_history", [])

    # Format chat history
    history_text = ""
    for i, message in enumerate(chat_history):
        role = "User" if message["role"] == "user" else "Assistant"
        history_text += f"{role}: {message['content']}\n"

    # Run QA
    response = qa_chain.run({
        "chat_history": history_text,
        "user_message": user_message
    })

    # Update history
    updated_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response}
    ]

    return {"bot_response": response, "chat_history": updated_history}

# Build the Chatbot Graph
graph = StateGraph(ChatState)
graph.add_node("start", start_node)
graph.add_node("answer", answer_node)

graph.add_edge("start", "answer")
graph.add_edge("answer", END)

graph.set_entry_point("start")

chatbot = graph.compile()
