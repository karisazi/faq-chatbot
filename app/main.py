import streamlit as st
import os
from chains import FAQChain
from faq_loader import FAQLoader
from utils import validate_query, clean_query
import time

def load_css(file_name: str):
    """Load external CSS file into Streamlit app (relative to this script)."""
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Streamlit page configuration
st.set_page_config(page_title="AXA Insurance FAQ Chatbot", layout="centered")

# Load external CSS
load_css("styles.css")

def create_streamlit_app(llm, faq, validate_query, clean_query):
    """
    Render the Streamlit chatbot app.
    Shows user text immediately, then bot replies after a short delay.
    """
    # Title + subtitle
    st.markdown("<div class='title'>🤖 AXA Insurance FAQ Assistant</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Powered by <a href='https://ai.google.dev/' target='_blank'>Google Gemini</a></div>",
        unsafe_allow_html=True
    )

    # Init chat history with initial greeting
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "bot", 
                "content": "Selamat datang di Asisten AXA! 👋\n\nSaya adalah asisten virtual yang siap membantu Anda dengan informasi seputar:\n• Pembayaran premi\n• Pengajuan klaim\n• Informasi polis\n• Produk asuransi AXA\n\nSilakan tanyakan apa saja yang ingin Anda ketahui. Saya akan dengan senang hati membantu Anda! 😊"
            }
        ]

    # Display chat history
    for msg in st.session_state.messages:
        css_class = "user-msg" if msg["role"] == "user" else "bot-msg"
        st.markdown(f"<div class='{css_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    # Clear chat button
    if st.button("🗑️ Hapus Percakapan", help="Klik untuk memulai percakapan baru"):
        st.session_state.messages = [
            {
                "role": "bot", 
                "content": "Selamat datang di Asisten AXA! 👋\n\nSaya adalah asisten virtual yang siap membantu Anda dengan informasi seputar:\n• Pembayaran premi\n• Pengajuan klaim\n• Informasi polis\n• Produk asuransi AXA\n\nSilakan tanyakan apa saja yang ingin Anda ketahui. Saya akan dengan senang hati membantu Anda! 😊"
            }
        ]
        st.rerun()

    # Chat input
    user_input = st.chat_input("Ketik pesan Anda...")
    if user_input:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Add temporary "thinking..." bot bubble
        st.session_state.messages.append({"role": "bot", "content": "💭 Thinking..."})

        st.rerun()

    # Replace "thinking..." with real answer (only if last bot message is placeholder)
    if st.session_state.messages and st.session_state.messages[-1]["content"] == "💭 Thinking...":
        # Grab last user message
        last_user_msg = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break

        if last_user_msg:
            # Validate & preprocess
            validated = validate_query(last_user_msg)
            if validated != last_user_msg.strip():
                st.session_state.messages[-1] = {"role": "bot", "content": validated}
                st.rerun()
                return

            cleaned_query = clean_query(validated)

            # Load FAQ + Context
            faq.load_faq()
            faq_context = faq.query_faq(cleaned_query)

            # Simulate "typing delay"
            time.sleep(1.5)

            # Generate final answer with chat history
            bot_answer = llm.generate_answer(cleaned_query, faq_context, st.session_state.messages)

            # Replace "thinking..." with real answer
            st.session_state.messages[-1] = {"role": "bot", "content": bot_answer}
            st.rerun()

if __name__ == "__main__":
    llm = FAQChain()
    faq = FAQLoader()
    create_streamlit_app(llm, faq, validate_query, clean_query)
