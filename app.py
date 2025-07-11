import streamlit as st
import os
from chatbot import load_chatbot
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Seward's CV Chatbot", page_icon="🤖")
st.title("👋 Hi there! I'm Seward's AI CV Assistant")
st.write("""I'm here to tell you all about Seward Mupereri's professional journey! 

🔍 Ask me about:
- Work experience and achievements
- Technical skills and expertise
- Education and certifications
- Or anything else you'd like to know!
""")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

script_dir = os.path.dirname(os.path.abspath(__file__))
cv_path = os.path.join(script_dir, 'cv.txt')

try:
    chain = load_chatbot(cv_path)

    if st.button("🔄 Restart Chat"):
        st.session_state.clear()
        st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about Seward's CV..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = chain({"question": prompt})["answer"]
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.chat_history = [(m["role"], m["content"]) for m in st.session_state.messages]

except FileNotFoundError as e:
    st.error(str(e))
