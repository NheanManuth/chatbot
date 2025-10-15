import os
import json
from typing import List, Dict, Any

import streamlit as st
from groq import Groq


# -----------------------------
# Utilities for persistence
# -----------------------------
def load_history_from_file(file_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, list):
                sanitized: List[Dict[str, str]] = []
                for item in data:
                    if isinstance(item, dict) and "role" in item and "content" in item:
                        sanitized.append({
                            "role": str(item["role"]),
                            "content": str(item["content"]),
                        })
                return sanitized
    except Exception:
        return []
    return []


def save_history_to_file(file_path: str, messages: List[Dict[str, str]]) -> None:
    try:
        serializable: List[Dict[str, str]] = [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in messages
        ]
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(serializable, file, ensure_ascii=False, indent=2)
    except Exception:
        pass


# -----------------------------
# Groq client handling
# -----------------------------
def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def stream_groq_completion(
    client: Groq,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int | None,
) -> str:
    """Stream a completion from Groq and return the full text."""
    streamed_text = ""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    # The stream yields chunks that include a delta with partial content
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                streamed_text += delta.content
                yield delta.content
        except Exception:
            # Best-effort streaming; ignore malformed chunks
            continue

    # Final return for completeness (not used by the generator consumer)
    return streamed_text


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Kon khmer ai", page_icon="💬", layout="centered")

st.title("💬Kon khmer ai")
st.caption("General-purpose assistant with conversation memory and optional local persistence.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")

    # API key management
    default_api_key = os.environ.get("GROQ_API_KEY", "")
    if "api_key" not in st.session_state:
        st.session_state.api_key = default_api_key

    api_key_input = st.text_input("GROQ API Key", value=st.session_state.api_key, type="password")
    if api_key_input and api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input

    # Model and generation controls
    model = st.selectbox(
        "Model",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ],
        index=0,
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    max_tokens_enabled = st.checkbox("Limit max tokens", value=False)
    max_tokens_value = st.number_input("Max tokens", min_value=1, max_value=8192, value=1024, step=64, disabled=not max_tokens_enabled)
    max_tokens = int(max_tokens_value) if max_tokens_enabled else None

    # System prompt (optional)
    system_prompt = st.text_area(
        "System prompt (optional)",
        value=st.session_state.get("system_prompt", "You are a helpful, concise assistant."),
        height=100,
    )
    st.session_state.system_prompt = system_prompt

    # Persistence controls
    st.subheader("Memory")
    persist_history = st.checkbox("Persist conversation to local JSON", value=False)
    history_file = st.text_input("History file path", value="chat_history.json", disabled=not persist_history)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        if persist_history and os.path.exists(history_file):
            try:
                os.remove(history_file)
            except Exception:
                pass


# Initialize conversation state
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

# Load from disk if persistence enabled and nothing loaded yet
if persist_history and not st.session_state.messages:
    loaded = load_history_from_file(history_file)
    st.session_state.messages = loaded

# Ensure a system prompt is present as the first message if provided and not already set
def ensure_system_prompt(messages: List[Dict[str, str]], system_content: str) -> List[Dict[str, str]]:
    if system_content.strip() == "":
        return messages
    if not messages or messages[0].get("role") != "system":
        return [{"role": "system", "content": system_content}] + messages
    messages[0]["content"] = system_content
    return messages

st.session_state.messages = ensure_system_prompt(st.session_state.messages, st.session_state.system_prompt)

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
user_input = st.chat_input("Ask me anything...")
if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare assistant placeholder
    with st.chat_message("assistant"):
        assistant_placeholder = st.empty()

        # Validate API key
        api_key = st.session_state.api_key.strip()
        if not api_key:
            assistant_placeholder.error("Please provide a GROQ API key in the sidebar.")
        else:
            client = get_groq_client(api_key)

            # Build the messages for the API call (include entire history)
            groq_messages: List[Dict[str, str]] = [
                {"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages
            ]

            # Stream response
            accumulated = ""
            try:
                for delta_text in stream_groq_completion(
                    client=client,
                    model=model,
                    messages=groq_messages,
                    temperature=float(temperature),
                    max_tokens=max_tokens,
                ):
                    accumulated += delta_text
                    assistant_placeholder.markdown(accumulated)
            except Exception as e:
                assistant_placeholder.error(f"Error while generating response: {e}")
                accumulated = ""

            # Append assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": accumulated})

            # Persist if enabled
            if persist_history:
                save_history_to_file(history_file, st.session_state.messages)


