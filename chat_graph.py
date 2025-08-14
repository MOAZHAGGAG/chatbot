from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import time
import tiktoken


class ChatState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], add_messages]
    model: str
    latency: float
    token_count: int
    streaming: bool


def llm_node(state: ChatState) -> Dict[str, Any]:
    from llm_node import process_message
    
    start_time = time.time()
    
    response, token_count = process_message(
        messages=state["messages"],
        model=state.get("model", "qwen3:4b"),
        streaming=state.get("streaming", True)
    )
    
    latency = time.time() - start_time
    
    return {
        "messages": [response],
        "latency": latency,
        "token_count": token_count
    }


def build_chat_graph():
    workflow = StateGraph(ChatState)
    
    workflow.add_node("llm", llm_node)
    
    workflow.add_edge(START, "llm")
    workflow.add_edge("llm", END)
    
    return workflow.compile()


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))