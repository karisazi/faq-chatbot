import streamlit as st
import os
from chains import FAQChain
from faq_loader import FAQLoader
from utils import validate_query, clean_query

def load_css(file_name: str):
    """Load external CSS file into Streamlit app (relative to this script)."""
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Streamlit page configuration
st.set_page_config(page_title="Chatbot FAQ Nawa - Nawatech", layout="centered")

# Load external CSS
load_css("styles.css")

def create_streamlit_app(llm, faq, validate_query, clean_query):
    """
    Render the Streamlit chatbot app.
    Handles chat history, input validation, FAQ retrieval, and LLM response generation.
    """
    # Title + subtitle
    st.markdown("<div class='title'>🤖 Chatbot FAQ Nawa</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Powered by <a href='https://www.nawatech.co/' target='_blank'>Nawatech</a></div>",
        unsafe_allow_html=True
    )

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        css_class = "user-msg" if msg["role"] == "user" else "bot-msg"
        st.markdown(f"<div class='{css_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Type your message...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})

    # Input validation
    validated = validate_query(user_input)
    if validated != user_input.strip():
        st.session_state.messages.append({"role": "bot", "content": validated})
        st.rerun()
        return

    # Preprocess query
    cleaned_query = clean_query(validated)

    # Load FAQ + Context Retrieval
    faq.load_faq()
    faq_context = faq.query_faq(cleaned_query)

    # Generate LLM answer
    bot_answer = llm.generate_answer(cleaned_query, faq_context)

    st.session_state.messages.append({"role": "bot", "content": bot_answer})
    st.rerun()

if __name__ == "__main__":
    llm = FAQChain()
    faq = FAQLoader()
    create_streamlit_app(llm, faq, validate_query, clean_query)
