"""
OpenAI LLM Node for Phase 2
This module handles OpenAI API integration with automatic cost tracking
"""

from typing import Dict, Any, List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def stream_openai_response(
    messages: List[Dict[str, Any]], 
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7
):
    """
    Stream OpenAI responses chunk by chunk
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: OpenAI model to use
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
    
    Yields:
        String chunks of the response
    """
    # Initialize the OpenAI chat model with LangChain
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=True
    )
    
    # Convert messages to LangChain format
    langchain_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            langchain_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
        else:
            langchain_messages.append(HumanMessage(content=content))
    
    try:
        # Stream the response
        for chunk in llm.stream(langchain_messages):
            if chunk and hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
    except (GeneratorExit, StopIteration):
        # Handle stream interruption gracefully
        pass
    except Exception as e:
        # Log error but don't crash
        yield f"\n*Error during streaming: {str(e)}*"


def process_openai_message(
    messages: List[Dict[str, Any]], 
    model: str = "gpt-3.5-turbo",
    streaming: bool = False,  # Default to False for non-streaming
    temperature: float = 0.7
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process messages using OpenAI's API via LangChain
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: OpenAI model to use (gpt-3.5-turbo, gpt-4, etc.)
        streaming: Whether to stream the response
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
    
    Returns:
        Tuple of (response_dict, metadata_dict)
        - response_dict: Contains role and content
        - metadata_dict: Contains tokens, cost, and model info
    """
    
    # Initialize the OpenAI chat model with LangChain
    # The API key is automatically read from OPENAI_API_KEY env variable
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=streaming
    )
    
    # Convert messages to LangChain format
    langchain_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Map roles to LangChain message types
        if role == "system":
            langchain_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
        else:  # user or any other role
            langchain_messages.append(HumanMessage(content=content))
    
    try:
        # NOTE: This function only handles non-streaming now
        # For streaming, use stream_openai_response() instead
        if streaming:
            raise ValueError("For streaming, use stream_openai_response() function instead")
            
        # For non-streaming, invoke the model directly
        response = llm.invoke(langchain_messages)
        full_response = response.content
        
        # Extract metadata from the response
        # LangChain's ChatOpenAI includes usage metadata when available
        metadata = {
            "model": model,
            "streaming": False
        }
        
        # Check if response has usage_metadata (available in latest versions)
        if hasattr(response, 'response_metadata'):
            usage = response.response_metadata.get('token_usage', {})
            metadata.update({
                "tokens_input": usage.get('prompt_tokens', 0),
                "tokens_output": usage.get('completion_tokens', 0),
                "total_tokens": usage.get('total_tokens', 0),
            })
            
            # Calculate cost based on OpenAI pricing
            cost = calculate_openai_cost(
                model=model,
                input_tokens=metadata['tokens_input'],
                output_tokens=metadata['tokens_output']
            )
            metadata['cost'] = cost
        else:
            # Fallback if metadata not available
            metadata.update({
                "tokens_input": None,
                "tokens_output": None,
                "total_tokens": None,
                "cost": None
            })
        
        # Return response and metadata
        return {"role": "assistant", "content": full_response}, metadata
            
    except Exception as e:
        # Handle errors gracefully
        error_msg = f"OpenAI API error: {str(e)}"
        return {"role": "assistant", "content": error_msg}, {
            "error": str(e),
            "model": model,
            "streaming": streaming
        }


def calculate_openai_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost of an OpenAI API call based on token usage
    
    Pricing as of 2025 (in USD per 1K tokens):
    - GPT-4o-mini: $0.00015 input, $0.0006 output (cheapest!)
    - GPT-3.5-turbo: $0.0005 input, $0.0015 output
    
    Note: GPT-4o-mini is 60%+ cheaper than GPT-3.5-turbo
    """
    
    # Define pricing per 1000 tokens (in USD)
    # Prices confirmed from OpenAI pricing page (2025)
    pricing = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # $0.15/$0.60 per million tokens
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},  # $0.50/$1.50 per million tokens
        "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},  # Same as gpt-3.5-turbo
        "gpt-3.5-turbo-1106": {"input": 0.0005, "output": 0.0015},  # Same as gpt-3.5-turbo
        # Keeping other models for reference only
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
    }
    
    # Get pricing for the model (default to GPT-3.5 pricing if model not found)
    model_pricing = pricing.get(model, pricing["gpt-3.5-turbo"])
    
    # Calculate cost
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    total_cost = input_cost + output_cost
    
    return round(total_cost, 6)  # Round to 6 decimal places for precision


def get_available_openai_models() -> List[str]:
    """
    Return the cheapest OpenAI model for the workshop
    
    GPT-4o-mini: The most cost-effective choice
    - 60%+ cheaper than GPT-3.5-turbo
    - 128K context window
    - Up to 16K output tokens
    - Knowledge cutoff: October 2023
    - Better performance than GPT-3.5-turbo
    """
    return [
        "gpt-4o-mini",   # Cheapest model: $0.15/$0.60 per million tokens
    ]


def check_openai_api_key() -> bool:
    """
    Check if OpenAI API key is configured
    """
    api_key = os.getenv("OPENAI_API_KEY")
    return api_key is not None and api_key != "your_openai_api_key_here" and len(api_key) > 0