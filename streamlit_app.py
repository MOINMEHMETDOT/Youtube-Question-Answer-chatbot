import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"

st.title("🎥 YouTube Q&A Assistant")

youtube_url = st.text_input(
    "Enter YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=Fa8JpMkz2HY"
)

if st.button("Process Video"):
    if not youtube_url:
        st.warning("Please enter a YouTube URL")
    else:
        with st.spinner("Processing video..."):
            res = requests.post(
                f"{API_BASE_URL}/youtube/ingest",
                json={"youtube_url": youtube_url}
            )

        if res.status_code == 200:
            st.success("Video processed successfully")
        else:
            st.error(res.text)

st.divider()

question = st.text_input("Ask a question")

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            res = requests.post(
                f"{API_BASE_URL}/youtube/ask",
                json={"question": question}
            )

        if res.status_code == 200:
            answer = res.json().get("answer")
            st.markdown("### Answer")
            st.write(answer)
        else:
            st.error(res.text)
