import streamlit as st
import time
from dotenv import load_dotenv
from chat_graph import count_tokens
from openai_node import (
    check_openai_api_key,
    process_openai_message,
    stream_openai_response
)

# Load environment variables
load_dotenv()

@st.cache_data
def load_college_info():
    try:
        with open("info.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "College information is currently unavailable."

st.set_page_config(
    page_title="Helwan Commerce College Chatbot",
    page_icon="🎓",
    layout="wide"
)

# Main header
st.markdown("""
<div style='background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem; text-align: center; color: white;'>
    <h1>🎓 Faculty of Commerce & Business Administration</h1>
    <p>Helwan University - Chatbot</p>
    <p style='font-size: 1rem; margin-top: 0.5rem;'>Ask anything about the college!</p>
</div>
""", unsafe_allow_html=True)

college_info = load_college_info()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_latency" not in st.session_state:
    st.session_state.total_latency = 0.0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

base_system_message = f"""You are an intelligent assistant for the Faculty of Commerce and Business Administration at Helwan University. Your mission is to help students and answer all their questions about the college.

College Information:
{college_info}

Important Instructions:
1. If the user's question is in Arabic, always reply in Arabic. If the user's question is in English, reply in English.
2. Be creative, generative, and helpful in your responses. Don't just copy information; synthesize, elaborate, and provide new insights, examples, and advice.
3. Use the information provided above as a knowledge base, but feel free to expand, explain, and add value beyond what is written.
4. If you cannot find an answer in the available information, politely apologize and suggest contacting the college administration.
5. Be friendly and helpful.
6. If a student asks about something outside the scope of the college, politely redirect them to ask about college-related topics.
"""

if "system_message" not in st.session_state:
    st.session_state.system_message = base_system_message

# Check OpenAI API key
if not check_openai_api_key():
    st.error("⚠️ OpenAI API key is not configured! Please add it to your .env file.")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message:
            cols = st.columns(3)
            with cols[0]:
                st.caption(f"🔢 {message['metadata']['tokens']} tokens")
            with cols[1]:
                st.caption(f"⏱️ {message['metadata']['latency']:.2f}s")
            with cols[2]:
                cost = message['metadata'].get('cost', 0)
                st.caption(f"💰 ${cost:.6f}" if cost > 0 else "💰 Free")

# Welcome message if no messages
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""
        ### Welcome to Helwan Commerce College Chatbot! 🎓
        
        Hello! I am your assistant for the Faculty of Commerce and Business Administration at Helwan University. 
        Ask me anything about:
        - College departments and specializations
        - Admission requirements and procedures
        - Tuition fees and financial information
        - Student activities and organizations
        - Internship and training opportunities
        - Career guidance and job prospects
        - Graduate studies information
        - Campus facilities and services
        - Academic support and counseling
        - Student housing information
        - Contact information
        
        What would you like to know?
        """)

# Chat input
if prompt := st.chat_input("Type your question here... 💬"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    user_tokens = count_tokens(prompt)
    st.session_state.total_tokens += user_tokens
    with st.chat_message("assistant"):
        start_time = time.time()
        try:
            messages_with_system = [
                {"role": "system", "content": st.session_state.system_message}
            ]
            recent_messages = st.session_state.messages[-6:]
            messages_with_system.extend(recent_messages)
            response_placeholder = st.empty()
            full_response = ""
            for chunk in stream_openai_response(
                messages=messages_with_system,
                model="gpt-4o-mini",
                temperature=0.7
            ):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            response_tokens = count_tokens(full_response) if full_response else 0
            from openai_node import calculate_openai_cost
            estimated_cost = calculate_openai_cost(
                model="gpt-4o-mini",
                input_tokens=user_tokens,
                output_tokens=response_tokens
            ) if response_tokens > 0 else 0
            metadata = {
                "tokens": response_tokens,
                "cost": estimated_cost,
                "streaming": True
            }
        except Exception as e:
            st.error(f"❌ OpenAI Error: {e}")
            full_response = "Sorry, I couldn't process your request."
            metadata = {"tokens": 0, "cost": 0, "error": str(e)}
            st.markdown(full_response)
        latency = time.time() - start_time
        cols = st.columns(3)
        with cols[0]:
            st.caption(f"🔢 {metadata.get('tokens', 0)} tokens")
        with cols[1]:
            st.caption(f"⏱️ {latency:.2f}s")
        with cols[2]:
            cost = metadata.get('cost', 0)
            st.caption(f"💰 ${cost:.6f}" if cost > 0 else "💰 Free")
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "metadata": {
                "tokens": metadata.get('tokens', 0),
                "latency": latency,
                "model": "gpt-4o-mini",
                "cost": metadata.get('cost', 0)
            }
        })
        st.session_state.total_tokens += metadata.get('tokens', 0)
        if metadata.get('cost'):
            st.session_state.total_cost += metadata['cost']
        if len(st.session_state.messages) > 2:
            st.session_state.total_latency = (
                st.session_state.total_latency * (len(st.session_state.messages) // 2 - 1) + latency
            ) / (len(st.session_state.messages) // 2)
        else:
            st.session_state.total_latency = latency
        st.rerun()

