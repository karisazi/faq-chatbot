import streamlit as st
import os
import sys
from chains import FAQChain, logger
from faq_loader import FAQLoader
from utils import validate_query, clean_query
import time

def load_css(file_name: str):
    """Load external CSS file into Streamlit app (relative to this script)."""
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def init_faq_system(force_reload=False, enable_warmup=False):
    """
    Initialize FAQ system with caching to prevent reloading on every Streamlit rerun.
    
    This is cached using @st.cache_resource so:
    - First call: Loads embeddings, connects to DB (~20-40s)
    - Subsequent calls: Returns cached instance (instant!)
    
    OPTIMIZATIONS:
    - Cached initialization (only loads once)
    - Optional cache warm-up for common queries
    
    :param force_reload: Force rebuild vector database
    :param enable_warmup: Pre-cache common queries for instant responses
    :return: Tuple of (FAQChain, FAQLoader)
    """
    print("🔧 Initializing FAQ system (this only happens once)...")
    
    # Initialize FAQ loader with dual collections
    faq = FAQLoader()
    
    # Load data with optional reload
    faq.load_faq(force_reload=force_reload)
    
    # Initialize multi-agent RAG system
    llm = FAQChain(faq_loader=faq)
    
    # OPTIMIZATION: Warm up cache with common queries (optional)
    if enable_warmup:
        print("🔥 Warming up cache with common queries...")
        llm.warm_up_cache()
    
    print("✅ FAQ system initialized and cached!")
    return llm, faq

# Streamlit page configuration
st.set_page_config(page_title="AXA Insurance FAQ Chatbot", layout="centered")

# Load external CSS
load_css("styles.css")

def create_streamlit_app():
    """
    Render the Streamlit chatbot app with optimized flow.
    
    OPTIMIZATIONS:
    - Cached system initialization (only loads once)
    - Process queries immediately without intermediate reruns
    - Use st.spinner() instead of time.sleep()
    - Single rerun per user message
    """
    # Initialize FAQ system (cached - only loads once!)
    llm, faq = init_faq_system()
    
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
                "content": "Selamat datang di Asisten AXA! 👋\n\nSaya dapat membantu Anda dengan informasi seputar:\n\n• Pembayaran premi (Informasi lebih lanjut bisa ditanyakan)\n• Pengajuan klaim (Informasi lebih lanjut bisa ditanyakan)\n• Informasi polis (Informasi lebih lanjut bisa ditanyakan)\n• Produk asuransi AXA, seperti Asuransi Perjalanan SmartTravel Domestik dan Internasional yang dapat dipilih di AXA myPage\n• Direktori Layanan Nasabah AXA, untuk mencari informasi lengkap mengenai kantor cabang, daftar rekanan rumah sakit, dan bengkel AXA melalui laman https://axa.co.id/direktori\n• Menghubungi Customer Care Center AXA Insurance untuk menanyakan klaim yang telah diajukan\n\nSilakan tanyakan apa saja yang ingin Anda ketahui. Saya akan dengan senang hati membantu Anda! 😊"
            }
        ]

    # Display chat history
    for msg in st.session_state.messages:
        css_class = "user-msg" if msg["role"] == "user" else "bot-msg"
        st.markdown(f"<div class='{css_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    # Floating clear chat button (top-right corner)
    st.markdown("""
    <style>
    .floating-button {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        background: #ff4444;
        color: white;
        border: none;
        border-radius: 50px;
        padding: 10px 15px;
        font-size: 14px;
        cursor: pointer;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .floating-button:hover {
        background: #cc0000;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Clear chat button - floating style
    if st.button("🗑️ Hapus", help="Klik untuk memulai percakapan baru", key="clear_chat"):
        st.session_state.messages = [
            {
                "role": "bot", 
                "content": "Selamat datang di Asisten AXA! 👋\n\nSaya dapat membantu Anda dengan informasi seputar:\n• Pembayaran premi (Informasi lebih lanjut bisa ditanyakan)\n• Pengajuan klaim (Informasi lebih lanjut bisa ditanyakan)\n• Informasi polis (Informasi lebih lanjut bisa ditanyakan)\n• Produk asuransi AXA, seperti Asuransi Perjalanan SmartTravel Domestik dan Internasional yang dapat dipilih di AXA myPage\n• Direktori Layanan Nasabah AXA, untuk mencari informasi lengkap mengenai kantor cabang, daftar rekanan rumah sakit, dan bengkel AXA melalui laman https://axa.co.id/direktori\n• Menghubungi Customer Care Center AXA Insurance untuk menanyakan klaim yang telah diajukan\n\nSilakan tanyakan apa saja yang ingin Anda ketahui. Saya akan dengan senang hati membantu Anda! 😊"
            }
        ]
        st.rerun()

    # Chat input - OPTIMIZED: Process immediately without intermediate rerun
    user_input = st.chat_input("Ketik pesan Anda...")
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Validate & preprocess
        validated = validate_query(user_input)
        if validated != user_input.strip():
            # Invalid input - show error
            st.session_state.messages.append({"role": "bot", "content": validated})
            st.rerun()
            return
        
        cleaned_query = clean_query(validated)
        
        # Generate answer with spinner (no time.sleep needed!)
        with st.spinner("🤔 Sedang berpikir..."):
            # Generate answer using multi-agent RAG system
            bot_answer = llm.generate_answer(cleaned_query, None, st.session_state.messages)
        
        # Add bot answer to history
        st.session_state.messages.append({"role": "bot", "content": bot_answer})
        
        # Single rerun to display new messages
        st.rerun()

def prompt_reload_vectordb():
    """
    Prompt user if they want to reload the vector database.
    Only shown when running via streamlit command (not when importing).
    """
    # Check if running in streamlit by checking for _is_running_with_streamlit
    try:
        import streamlit.runtime.scriptrunner.script_runner
        # If this succeeds, we're running in streamlit - don't prompt
        return False
    except:
        pass
    
    # Check if vectorstore exists
    vectorstore_path = os.path.join(os.path.dirname(__file__), 'vectorstore')
    if not os.path.exists(vectorstore_path) or not os.listdir(vectorstore_path):
        print("\n📦 No existing vector database found. Will create new one.")
        return False
    
    # Prompt user
    print("\n" + "="*60)
    print("🗄️  VECTOR DATABASE RELOAD")
    print("="*60)
    print("Existing vector database found!")
    print("  Location: ./vectorstore/")
    print("\nOptions:")
    print("  [1] Use existing (fast - skip embedding)")
    print("  [2] Rebuild database (slow - re-embed all data)")
    print("  [3] Delete and exit (manual cleanup)")
    print("="*60)
    
    while True:
        choice = input("\nYour choice [1/2/3]: ").strip()
        
        if choice == '1':
            print("✓ Using existing vector database\n")
            return False
        elif choice == '2':
            print("✓ Will rebuild vector database (this may take time...)\n")
            return True
        elif choice == '3':
            print("✓ Exiting. To delete, run: rm -rf vectorstore/\n")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    # OPTIMIZED: Initialization now happens inside @st.cache_resource
    # This prevents reloading on every Streamlit rerun!
    
    # Check if we should reload vector DB (only when not in streamlit)
    # Note: force_reload is not yet supported with caching
    # To force reload, clear Streamlit cache: streamlit cache clear
    if len(sys.argv) > 1 and sys.argv[1] == '--reload':
        print("🔄 Force reload flag detected")
        print("⚠️  To force reload with cached version, run: streamlit cache clear")
    
    # Just run the app - initialization happens inside via @st.cache_resource
    create_streamlit_app()
