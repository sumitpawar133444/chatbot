from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from .config import llm

# Chatbot State
class ChatState(dict):
    pass

# Prompt Template
prompt = PromptTemplate(
    input_variables=["user_message"],
    template="You are a helpful AI assistant.\n\nUser: {user_message}\n\nAssistant:"
)

# Simple LangChain QA Chain
qa_chain = LLMChain(llm=llm, prompt=prompt)

# Graph Nodes
def start_node(state):
    return {"user_message": state["user_message"]}

def answer_node(state):
    user_message = state["user_message"]
    bot_response = qa_chain.run(user_message)
    return {"bot_response": bot_response}

# Build the Chatbot Graph
graph = StateGraph(ChatState)
graph.add_node("start", start_node)
graph.add_node("answer", answer_node)

graph.add_edge("start", "answer")
graph.add_edge("answer", END)

graph.set_entry_point("start")

# Compiled chatbot
chatbot = graph.compile()
