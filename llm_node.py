import ollama
from typing import Dict, Any, List, Tuple, Generator
import streamlit as st


def process_message(
    messages: List[Dict[str, Any]], 
    model: str = "qwen3:4b",
    streaming: bool = True
) -> Tuple[Dict[str, Any], int]:
    
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            formatted_messages.append(msg)
        else:
            formatted_messages.append({
                "role": getattr(msg, "type", "user"),
                "content": getattr(msg, "content", str(msg))
            })
    
    if streaming and "stream_container" in st.session_state:
        response_text = ""
        with st.session_state.stream_container:
            message_placeholder = st.empty()
            
            stream = ollama.chat(
                model=model,
                messages=formatted_messages,
                stream=True
            )
            
            for chunk in stream:
                if chunk.get("message", {}).get("content"):
                    response_text += chunk["message"]["content"]
                    message_placeholder.markdown(response_text)
        
        token_count = estimate_tokens(response_text)
        return {"role": "assistant", "content": response_text}, token_count
    
    else:
        try:
            response = ollama.chat(
                model=model,
                messages=formatted_messages,
                stream=False
            )
            
            # Handle different response formats from Ollama v0.5.x
            if isinstance(response, dict):
                if "message" in response:
                    response_text = response["message"]["content"]
                elif "content" in response:
                    response_text = response["content"]
                else:
                    response_text = str(response)
            else:
                # Handle new response object format
                response_text = getattr(response, 'message', {}).get('content', str(response))
                
            token_count = estimate_tokens(response_text)
            return {"role": "assistant", "content": response_text}, token_count
            
        except Exception as e:
            print(f"Error in non-streaming chat: {e}")
            # Return error message
            error_msg = f"Error processing request: {str(e)}"
            return {"role": "assistant", "content": error_msg}, estimate_tokens(error_msg)


def stream_ollama_response(
    messages: List[Dict[str, Any]], 
    model: str = "qwen3:4b"
) -> Generator[str, None, None]:
    
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            formatted_messages.append(msg)
        else:
            formatted_messages.append({
                "role": getattr(msg, "type", "user"),
                "content": getattr(msg, "content", str(msg))
            })
    
    stream = ollama.chat(
        model=model,
        messages=formatted_messages,
        stream=True
    )
    
    for chunk in stream:
        if chunk.get("message", {}).get("content"):
            yield chunk["message"]["content"]


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)  


def get_available_models() -> List[str]:
    try:
        response = ollama.list()
        
        # Handle the new ollama API (v0.5.x) which returns a ListResponse object
        if hasattr(response, 'models'):
            # Extract model names from the Model objects
            model_names = [model.model for model in response.models]
            return model_names if model_names else ["qwen3:4b", "llama3.2", "mistral"]
        
        # Fallback for older API versions
        elif isinstance(response, dict) and 'models' in response:
            models = response['models']
            model_names = []
            for model in models:
                if isinstance(model, dict):
                    model_name = model.get('name', model.get('model', None))
                    if model_name:
                        model_names.append(model_name)
            return model_names if model_names else ["qwen3:4b", "llama3.2", "mistral"]
        
        else:
            return ["qwen3:4b", "llama3.2", "mistral"]
            
    except Exception as e:
        print(f"Error getting models: {e}")
        return ["qwen3:4b", "llama3.2", "mistral"]