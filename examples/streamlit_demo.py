"""Streamlit front-end for the single-agent SEC Filing Analyzer demo."""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import humanize
import streamlit as st

# Import the build_demo_agent function directly
# We need to use an absolute import to avoid conflicts with src/examples
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_chat_demo import build_demo_agent


def main() -> None:
    os.environ["SFA_DEMO_MODE"] = "1"

    st.set_page_config(page_title="SEC Filing Analyzer", layout="wide")
    st.title("📊 SEC Filing Analyzer — Demo")

    agent = build_demo_agent()

    # Chat UI using Streamlit Experimental chat components (built-in v1.25+).
    if "history" not in st.session_state:
        st.session_state.history = []

    for msg, is_user in st.session_state.history:
        avatar = "👤" if is_user else "🤖"
        st.chat_message("user" if is_user else "assistant", avatar=avatar).markdown(msg)

    prompt = st.chat_input("Ask about an SEC filing…")
    if prompt:
        st.session_state.history.append((prompt, True))
        st.chat_message("user", avatar="👤").markdown(prompt)

        with st.spinner("Analyzing…"):
            answer = agent.respond(prompt)

        st.session_state.history.append((answer, False))
        st.chat_message("assistant", avatar="🤖").markdown(answer)


if __name__ == "__main__":
    main()
