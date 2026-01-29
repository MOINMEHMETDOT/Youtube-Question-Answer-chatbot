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

# ---------------- GLOBAL CACHES (RAM OPTIMIZATION - CRITICAL) ----------------
embeddings_cache = None
llm_cache = None

def get_embeddings():
    """Global embedding cache - loads ONLY ONCE"""
    global embeddings_cache
    if embeddings_cache is None:
        print("Loading embeddings model (one-time)...")
        embeddings_cache = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        print("Embeddings loaded!")
    return embeddings_cache

def get_llm():
    """Global LLM cache - loads ONLY ONCE"""
    global llm_cache
    if llm_cache is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        print("Initializing LLM (one-time)...")
        llm_cache = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-exp",  # Lighter model
            temperature=0.7,
            api_key=api_key
        )
        print("LLM ready!")
    return llm_cache

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
    return "\n\n".join(doc.page_content for doc in docs)  # Fixed: proper newline

# ---------------- ULTIMATE BOT-PROOF Transcript Fetch ----------------
def get_transcript_with_ytdlp(video_id: str) -> str:
    """ULTIMATE FIX: YouTube bot detection bypass + multiple fallbacks"""
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "skip": "hls,dash,live,translations",
                "player_skip": "js",
                "http_requests": "no-redirect"
            }
        },
    }
    
    try:
        print(f"Fetching transcript for video: {video_id}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://youtube.com/watch?v={video_id}", download=False)
        
        # TRY ALL SUBTITLE SOURCES (ULTIMATE FALLBACK)
        for sub_type in ["subtitles", "automatic_captions"]:
            subs = info.get(sub_type, {})
            if "en" in subs:
                try:
                    subtitle_url = subs["en"][0]["url"]
                    print(f"Trying {sub_type} subtitles...")
                    response = requests.get(subtitle_url, timeout=30)
                    response.raise_for_status()
                    
                    lines = response.text.splitlines()
                    text_lines = [
                        line for line in lines
                        if line and not line.startswith(("WEBVTT", "Kind:", "Language:", "00:"))
                    ]
                    transcript = " ".join(text_lines)
                    if len(transcript.strip()) > 100:  # Valid transcript
                        print(f"✅ Success! Transcript length: {len(transcript)} chars")
                        return transcript
                except Exception as sub_error:
                    print(f"Subtitle fetch failed ({sub_type}): {sub_error}")
                    continue
        
        raise RuntimeError("No English subtitles available after all attempts")
        
    except Exception as e:
        print(f"❌ Full failure: {str(e)}")
        raise RuntimeError(f"Transcript fetch failed: {str(e)}")

# ---------------- Core Logic ----------------
def build_rag_chain(youtube_url: str):
    """
    Input: YouTube URL
    Output: Runnable RAG chain (RAM optimized)
    """
    # 1️⃣ Normalize URL → video_id
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError("Could not extract video ID from URL")

    print(f"Processing video: {video_id}")

    # 2️⃣ Transcript (ULTIMATE BOT-PROOF)
    transcript = get_transcript_with_ytdlp(video_id)

    # 3️⃣ Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.create_documents([transcript])
    print(f"Created {len(chunks)} chunks")

    # 4️⃣ Embeddings + Vector Store (GLOBAL CACHE)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 5️⃣ LLM (GLOBAL CACHE)
    llm = get_llm()

    # 6️⃣ Prompt
    prompt = PromptTemplate(
        template="""
Answer ONLY using the context below. Be concise and insightful.
If the answer is not present, say "Not found in video transcript".

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    # 7️⃣ Chain
    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ RAG chain built successfully!")
    return chain
