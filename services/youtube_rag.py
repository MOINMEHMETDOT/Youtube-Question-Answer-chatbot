import os
import yt_dlp
import requests
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()


# ---------------- Utility ----------------
def extract_video_id(youtube_url: str) -> str:
    parsed = urlparse(youtube_url)

    # https://www.youtube.com/watch?v=VIDEO_ID
    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [None])[0]

    # https://youtu.be/VIDEO_ID
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")

    raise ValueError("Invalid YouTube URL")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------- Transcript via yt-dlp ----------------
def get_transcript_with_ytdlp(video_id: str) -> str:
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(
            f"https://www.youtube.com/watch?v={video_id}",
            download=False
        )

    subtitles = info.get("subtitles") or info.get("automatic_captions")
    if not subtitles or "en" not in subtitles:
        raise RuntimeError("No English subtitles available")

    subtitle_url = subtitles["en"][0]["url"]
    response = requests.get(subtitle_url)
    response.raise_for_status()

    lines = response.text.splitlines()
    text_lines = [
        line for line in lines
        if line
        and not line.startswith(("WEBVTT", "Kind:", "Language:", "00:"))
    ]

    return " ".join(text_lines)


# ---------------- Core Logic ----------------
def build_rag_chain(youtube_url: str):
    """
    Input: YouTube URL
    Output: Runnable RAG chain
    """

    # 🔐 API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    # 1️⃣ Normalize URL → video_id
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError("Could not extract video ID from URL")

    # 2️⃣ Transcript (yt-dlp, cloud-safe)
    transcript = get_transcript_with_ytdlp(video_id)

    # 3️⃣ Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.create_documents([transcript])

    # 4️⃣ Embeddings + Vector Store
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 5️⃣ LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=api_key
    )

    # 6️⃣ Prompt
    prompt = PromptTemplate(
        template="""
Answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    # 7️⃣ Chain
    chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    return chain | prompt | llm | StrOutputParser()
